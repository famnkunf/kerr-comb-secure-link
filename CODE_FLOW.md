# How the simulation flows, in plain language

This document explains what [main.ipynb](./main.ipynb) does, in the order the code
runs, and *why* each step is where it is. It was written to accompany the review
fixes; the previous version of the notebook is kept as `main.ipynb.bak`.

The notebook is a five-part story:

| Cell | Id | What it does | Run time |
|---|---|---|---|
| 0 | `5dec8627` | Builds the optical frequency comb (the light source). | ~4 s |
| 1 | `74e1985e` | The secure link library: keys, scrambler, channel, receiver, metrics, one end-to-end frame. | ~0.2 s |
| 2 | `405f2067` | Empty spacer. | – |
| 3 | `02b0b937` | Pictures of one frame as it travels through the link, plus the carrier-offset detector. | ~1 s |
| 4 | `7643339a` | Hopping the data across comb lines, and what a wideband eavesdropper can do. | ~1 s |
| 5 | `0153548f` | Bit-error-rate against OSNR, and the measured cost of the scrambling. | ~6 s |

Everything runs in about 10–12 seconds in one Python 3 kernel with NumPy and
Matplotlib. Nothing here is a fibre-optic link simulator: it is a
signal-processing and key-management model that runs on a computer. Where the
model stops being physical, the code says so in a comment and this document says
so again in the last section.

---

## 1. The words used in the code

* **Symbol** – one QAM-16 point, carrying 4 bits. The constellation is a 4x4 grid
  at ±1 and ±3 on each axis, scaled so the average power is 1.
* **Dwell** – a run of consecutive symbols sent on one comb line before hopping
  to the next line.
* **Frame** – one block of transmitted payload, 4096 or 8192 symbols here.
* **Comb line / mode (µ)** – one spectral tooth of the frequency comb. Each tooth
  is a separate carrier that the transmitter can put data on.
* **CFO** – carrier frequency offset, the unknown frequency difference between
  Alice's and Bob's lasers. This is what the receiver must estimate blindly.
* **OSNR** – optical signal-to-noise ratio, quoted in a 0.1 nm reference
  bandwidth (12.5 GHz at 1550 nm), as optical engineers do.
* **Es/N0** – the energy per symbol to noise ratio the receiver actually sees. The
  two are related by `Es/N0 = OSNR − 10·log10(symbol rate / 12.5 GHz)`.
* **SER / BER** – symbol error rate and bit error rate. For Gray-coded 16-QAM,
  `BER ≈ SER/4` at high SNR.
* **SD-FEC threshold** – the pre-FEC bit error rate the soft-decision forward
  error correction can still clean up. Set to 2e-2 (2%) here, i.e. an SER of
  about 8e-2.
* **Pilot** – a short known sequence inserted so the receiver can find its phase.

---

## 2. Cell 0 — where the light comes from

The comb is produced by the **Lugiato–Lefever equation (LLE)**, which describes
the light circulating inside a tiny ring resonator driven by a continuous pump
laser. The equation says, in words: the field loses energy, it shifts in phase
depending on how far the laser is detuned from the cavity resonance, it spreads
out through dispersion, it pushes on itself through the Kerr effect, and the pump
keeps topping it up.

The solver is a **split-step Fourier method**: at each time step it applies the
nonlinear (Kerr) part in the time domain and the linear (dispersion, detuning,
pump) part in the frequency domain, where the linear part is exactly solvable.

Then a **laser scan** runs: the detuning is swept from the blue side of resonance
to the red side, which is how a real experiment walks the resonator through
modulational instability into the soliton regime. The four panels show the
intracavity intensity profile, the resulting comb spectrum, the space–time
evolution, and the power-versus-detuning curve with the analytic bistability
branches drawn on top for comparison.

**New in this version (three things):**

1. The analytic bistability curves used to be evaluated at `rho = 0`, which
   divided by zero and produced the `RuntimeWarning: divide by zero` seen in the
   old output. They now start at `rho = 0.02`.
2. The scan is a fast transient, so after it finishes the code now **holds the
   final detuning** for 100 more normalised time units and then **classifies the
   steady state** (counts localised peaks in intensity and reports
   "homogeneous CW", "single soliton" or "N-soliton crystal"). It never assumes
   the answer. With the seed used here the answer is a **2-soliton crystal**, and
   the pump line still holds 35% of the power. Holding longer or changing the
   seed changes the count — that sensitivity is exactly why the classifier
   exists.
3. It **measures the per-line power profile** of the resulting comb (in dB
   relative to the strongest comb line, pump line excluded) and exports it as
   `comb_line_power_db` / `comb_mode_index` / `comb_state` for the later cells.

This is the plumbing that was missing before: cell 4 now receives a *measured*
comb envelope instead of assuming that all 64 lines carry equal power. With the
2-soliton state here, the envelope is 51.8 dB deep across ±32 modes and only
**14 of 64 lines are within 6 dB of the strongest**, which is what the hop layer
can actually use.

It stops short of one thing: converting relative dB into absolute dBm needs the
device parameters (FSR, Q, coupling, pump power, Kerr nonlinearity, mode area),
which are not fixed here. The code says so rather than inventing numbers.

---

## 3. Cell 1, part A — keys and the scrambler

**One secret, many keys.** `HybridKeyEngine` takes a root secret (in the story,
from a QKD link or a physically unclonable function) and derives independent
256-bit sub-keys from it using HMAC-SHA256 with a text label such as `"PD-KEY"`
(payload scrambling) or `"HOP-KEY"` (which comb line to use). Same root, same
labels, same keys at both ends; nothing is transmitted.

**The keystream.** `keystream_uniforms` asks SHAKE-256 for a stream of bytes from
`key ‖ label ‖ frame_id ‖ block_id` and turns them into numbers between 0 and 1.
Three properties matter:

* The **whole** 32-byte key goes in, so the mask has 256 bits of entropy. The old
  code seeded NumPy's PCG64 from the first four bytes, i.e. 32 bits, which is
  brute-forceable, and used a non-cryptographic generator.
* The **frame_id** is mixed in, so every frame gets a different mask. Without it,
  the same mask repeats frame after frame, and knowing one frame's payload
  unmasks every later frame (cell 4 demonstrates this attack and shows the fix
  defeats it).
* It is a one-way function, so seeing the mask does not reveal the key.

**The mask for one block.** `scramble_plan` draws three things from that stream:

* a **permutation** of the block (sorting 1024 uniform numbers — exactly uniform,
  no shuffling loop),
* an **amplitude** `A` uniform in [0.6, 1.4],
* a **phase** uniform in [0, 2π),
* plus a 32-symbol **pilot** drawn from the same stream.

**Transmit stream layout.** Each 1024-symbol payload block becomes
`[32 keyed pilot symbols][1024 scrambled payload symbols]`, so a 4096-symbol frame
is 4224 samples — a 3.0% overhead. The pilot is *keyed* rather than a public
constant on purpose: an eavesdropper cannot use it as known plaintext.

The transmitter also **interleaves**: payload symbol `perm[i]` is sent at time
`i` inside the block, then multiplied by `A·e^{jφ}`. The result looks like an
isotropic cloud — but see the honesty note in section 9, because it is not
perfectly isotropic.

---

## 4. Cell 1, part B — the channel

`optical_channel` applies, in order:

1. an **unknown carrier frequency offset** – a constant rotation at rate
   `cfo_hz`. The receiver is *never* given this number; that is the whole point
   of the change,
2. **laser phase noise** – a Wiener random walk with step
   `sqrt(2π·linewidth/symbol rate)`; the phase wanders slowly, and every comb
   line shares it because they come from the same pump,
3. **amplifier noise** – additive white Gaussian noise scaled so the electrical
   Es/N0 comes out as requested. You can specify either Es/N0 directly or an
   OSNR in 0.1 nm, which converts with `osnr_db_to_esn0_db`,
4. optionally, a **loss-of-signal burst** (`dropout=(start, stop)`), which blanks
   a stretch of samples to emulate the receiver losing lock.

There is no fibre here: no attenuation, dispersion, polarisation, Kerr
nonlinearity, amplifier-ASE accumulation or converter bandwidth limits. That is a
scope limitation, stated in the docstring.

---

## 5. Cell 1, part C — the receiver, and why the steps are in this order

This is the part that changed most. The order matters, so here it is with the
reasoning:

1. **De-scramble (in time order).** Only the payload samples are divided by
   `A·e^{jφ}`; the pilots are left alone. This needs only the key, not the
   frequency offset, so it can happen first. Crucially, it *must* happen before
   frequency estimation: while the mask is still on, the constellation is
   circularly symmetric, every frequency looks equally plausible, and the classic
   "fourth power" trick has nothing to lock onto. After de-scrambling we have an
   ordinary QAM-16 rotated by an unknown amount.
2. **Blind coarse CFO acquisition** (`acquire`). Raises the de-scrambled signal
   to the 4th power. For unit-magnitude QAM-16 points that has a non-zero mean,
   so its spectrum shows a line at four times the offset; the code reads off the
   top few candidate frequencies (`cfo_candidates`, capture range ±Rs/8 = ±4 GHz).
   Candidates cannot be judged on a raw stream — an unknown constant phase or a
   fast rotation both spoil any score — so **each candidate is applied and the
   phase recovery is run, and the candidate that actually locks wins**
   (`lock_quality`). The winning phase-corrected signal is returned so the work
   is not repeated.
3. **Carrier phase recovery** (`carrier_phase_recovery`). A sliding-window blind
   phase search: for each 32-symbol sub-block it tries 32 (or 64) candidate phase
   corrections within ±0.25 rad of the running estimate and keeps the one whose
   symbol decisions are least distant from the constellation. The window is
   deliberately narrower than a quadrant because QAM-16 maps onto itself under
   90° rotations — a full search has four equally good answers. Its tracking range
   is `dphi/(2π·sub_len)·Rs ≈ ±40 MHz`; anything faster must be removed by step 2
   first. This is why deleting the CFO estimator is not optional: the channel
   applies 1.2 GHz, which is 30× outside the phase tracker's pull-in range.
4. **Resolve the quadrant with the pilot** (`resolve_quadrant`). Blind phase
   recovery cannot tell which of the four 90° rotated copies of the
   constellation it locked to. Each block's known pilot gives an unambiguous
   phase; the code measures it, rounds to the nearest 90°, and de-rotates. This is
   what makes re-acquisition after an outage work.
5. **De-interleave** (`extract_payload`) — undo the permutation to get payload
   order back. Note this is *after* phase tracking on purpose: the frequency
   offset is a ramp in time, and permuting the symbols would scramble that ramp.

Everything above is wrapped in `decode_frame`, which returns all the intermediate
stages as a dictionary so cell 3 can plot them.

**Re-acquisition** (`detect_outage`, `reacquire_after_outage`): the receiver
watches the sub-block averaged power. If part of the frame goes quiet (a loss of
signal), it reports where the signal came back, skips forward to the next whole
block boundary, and runs the whole acquisition chain again on what is left. With
the pilot in place this re-locks with zero errors from the first symbol after the
outage.

---

## 6. Cell 1, part D — the numbers, and one honest limit

`error_stats` reports SER, an estimated BER (`SER/4`), and — when no error is seen
— the 95% upper bound `3/n` ("rule of three"). That last number is the honest
resolution of a run: **with 4096 symbols and zero errors you may only claim the
SER is below about 7e-4.** The old notebook's "SER = 0.0000e+00" said nothing at
all about how good the link was.

`fec_verdict` then compares the BER against the stated 2e-2 threshold and can
answer "PASS", "FAIL", or "INCONCLUSIVE — the measurement is fine but the upper
bound is not tight enough". The old code compared an SER against a hard-coded
`0.05` and labelled it "Decodable by SD-FEC"; that was neither the BER threshold
nor its SER equivalent (about 8e-2), so it has been replaced.

`scrambling_penalty` computes the analytic cost of the amplitude mask: the
receiver divides by `A`, so the noise is re-scaled by `1/A`, costing
`10·log10(E[1/A²]) = 0.76 dB`, or **0.98 dB net** once the extra launch power
(`E[A²] = 1.05`) is accounted for. The peak amplitude also rises, so PAPR goes
from 2.55 dB to 5.25 dB — a **+2.70 dB** headroom cost that would matter in a real
fibre. Cell 5 then *measures* the penalty and gets 1.20 dB.

Thresholds: SD-FEC 2e-2, OSNR reference bandwidth 12.5 GHz, symbol rate 32 GBd,
32-symbol pilot per 1024-symbol block.

---

## 7. Cell 3 — pictures of one frame

Six panels: the payload, the scrambled cloud, the de-scrambled signal still
rotating under the CFO, the same signal after blind CFO correction, the 4th-power
periodogram with the true offset (dashed) and the estimate (dotted) drawn on top,
and the recovered constellation with errors marked. In the run captured in the
notebook the detector puts the line at 4.802 GHz against a true 4×1.200 = 4.800
GHz and the frame decodes with zero errors.

---

## 8. Cell 4 — hopping across comb lines, and the eavesdropper

**The hop schedule.** The frame is chopped into **dwells** of 128 symbols
(4 ns at 32 GBd). A keyed permutation decides, dwell by dwell, which usable comb
line carries that dwell; every round visits each usable line exactly once, so
occupancy is uniform and no line is special. Both ends derive the schedule from
`HOP-KEY`, so nothing is signalled. Two assumptions are stated openly in the
code: (a) Bob's front end follows the schedule — a real radio needs a
comb-matched receiver bank or a tunable LO with guard intervals, and that cost is
not simulated; (b) the comb envelope is static and public, so both ends hold a
per-line calibration table (not key material) and divide it out.

**Per-line OSNR comes from cell 0.** Only lines that still clear the SD-FEC
threshold count as usable, so with the measured profile 14 of 64 lines are in
play. Their powers differ by 5.4 dB, and Bob's error rate is reported separately
for the strong and weak ends of that range.

**The eavesdropper is now the strong one.** She has a wideband coherent receiver
across the whole comb and knows her own noise floor, so she identifies the
occupied line in each dwell by comparing normalised dwell power. The notebook
shows three things about her:

* *Hopping alone buys nothing.* With no dummy traffic she identifies the right
  line **100%** of the time, so she captures every scrambled symbol. The security
  therefore rests entirely on the scrambler, not on the hopping.
* *Dummy traffic works.* When idle lines carry keyed dummy symbols with matched
  statistics, her hit rate collapses to **9.1%** against a 7.1% chance level. The
  price is printed too: filling 63 idle lines at full power is 18.1 dB of extra
  optical power, or the same power spread thinner.
* *What she can still do with a scrambled stream.* She slices it as if it were
  plain QAM-16 (SER 0.877 — near the 0.9375 chance level), and she runs a maximum
  likelihood estimate of the payload's **power class** from `|y|`. That works
  surprisingly well: 0.777 correct against a 0.500 baseline, worth **0.607
  bits/symbol out of 4 (15%)**. Her best symbol guess still fails (SER 0.857),
  because the phase mask destroys the quadrant information, but the amplitude
  leak is real and measurable — that is the "mutual information bound" the
  original notebook claimed without ever computing.

**The mask-reuse attack.** The notebook assumes one frame's payload is known
(the standard known-plaintext assumption: a header, a retransmission, a
predictable idle pattern) and uses it to learn the mask, then applies that mask
to the next frame. With a static mask the payload of every later frame falls out
with **SER 0.0000** — a total break. With the per-frame nonce introduced here the
same attack yields SER 0.9327, i.e. chance. That is the single most important
security fix in this revision, and it is now demonstrated rather than asserted.

---

## 9. Cell 5 — BER against OSNR, and the measured cost of the mask

The sweep runs 8 OSNR values × 2 seeds × 16384 symbols, with a fresh random
unknown CFO on every point, for two configurations: amplitude scrambling on (the
proposal) and off (phase-only scrambling, the reference). Pooled error counts give
the BER; points where nothing was observed are plotted as open triangles at their
95% upper bound instead of as "zero".

Results: at 26 dB OSNR the masked link sits at BER 3.1e-5; the error rate crosses
the 2e-2 SD-FEC threshold at **18.07 dB** with the mask and **16.87 dB** without
it. So the amplitude mask costs a measured **1.20 dB** (analytic 0.98 dB) — the
number the review asked for. Across the whole sweep the blind CFO estimate stays
within 0.14 MHz, i.e. the acquisition is not the weak link.

---

## 10. What changed, mapped to the review

| Review point | What the code does now | Still open |
|---|---|---|
| 1. NumPy RNG, 32-bit seed | SHAKE-256 keystream over the full 256-bit sub-key; mask separated per frame and per block; permutation, amplitude, phase and pilot all derived from it; a single `keystream_uniforms` used by every consumer (the hand-replayed RNG in the old hopping cell is gone). The mask-reuse attack in cell 4 shows the difference. | Nothing further is provable by simulation: only that no *stated* attack works. |
| 2. CFO handed to Bob; no re-acquisition | The channel draws the offset, the receiver never sees it. Blind acquisition from the 4th-power spectrum, each candidate verified by running the phase recovery, then pilot-based quadrant resolution, then de-interleaving. A loss-of-signal test re-acquires from the next block boundary with zero errors. | Pre-de-scramble estimation is impossible by construction, so the estimator is key-assisted; that is a design property, not a defect. |
| 3. Not a fibre; hand-set SNR; 4096 symbols | Proper OSNR↔Es/N0 bookkeeping, a BER-vs-OSNR waterfall against the stated 2e-2 threshold, the 95% resolution bound printed with every zero-error result, the FEC gate corrected (BER-based, not `SER < 0.05`), and the amplitude/PAPR penalty both computed (0.98 dB, +2.70 dB) and measured (1.20 dB). | No split-step fibre, no EDFA-ASE accumulation, no dual-polarisation, no 10⁶-symbol campaign, no CD/PMD equalisation, no converter limits. That needs a different tool. |
| 4. Per-symbol hopping; no sech² envelope | Dwell-based hopping (128 symbols per dwell, schedule keyed), per-line OSNR taken from cell 0's measured envelope, the usable-line count derived from the FEC threshold (14 of 64 here), and Bob's SER split by line strength. | Real hardware dwells of 1–100 µs need 32k–3.2M symbols per dwell and frames of ~10⁶ symbols, which this pure-Python DSP cannot run; the code says so and parameterises the dwell. |
| 5. Straw-man eavesdropper | A wideband receiver over the whole comb, per-dwell line identification, dummy traffic with matched statistics (A/B measured), a MAP attack on payload amplitude with its information leak in bits/symbol, and the mask-reuse attack. | No general proof exists that no better attack on the scrambler exists; the leak is quantified, not eliminated. |
| 6. Comb cell disconnected | The LLE state is classified instead of assumed, the per-line profile in dB is measured and exported, and cell 4 consumes it for per-line OSNR. Two bugs fixed (division by zero in the bistability curves, and the missing steady-state hold). | No single soliton: with these parameters the device settles into a 2-soliton crystal, and a single soliton decays. No absolute dBm: the device parameters are not fixed. No calibration against a measured spectrum. |

---

## 11. What this notebook still is not

* **It is not a fibre link.** No dispersion, no Kerr nonlinearity in the
  transmission fibre, no polarisation, no amplifier cascade, no converter
  bandwidth or quantisation. The "80 km" claim in the review comments cannot be
  supported by this code at all; that needs a split-step fibre model with EDFA
  ASE accumulation and about a million symbols per point.
* **It is not a security proof.** It shows that specific, named attacks fail once
  the mask is fresh per frame and derived from the full key. It cannot show that
  no other attack exists. In particular the amplitude leak (~0.6 bits/symbol
  here) is a property of using 16-QAM with a per-symbol mask; a constant-modulus
  constellation would not have it.
* **It is not a hardware validation.** Hop rates, switching times, comb
  flatness, LO tuning, guard intervals and calibration are modelled or assumed,
  never measured.
* **The comb numbers are relative.** Absolute dBm per line needs FSR, Q,
  coupling, pump power, nonlinearity and mode area; only the shape of the
  envelope is claimed here.

---

## 12. Running it and turning the knobs

Run the cells in order; cells 3, 4 and 5 depend on names defined in cells 0 and
1. Useful knobs, all near the top of their cell:

| Cell | Name | Meaning |
|---|---|---|
| 0 | `N`, `d2`, `f2`, `dtau` | LLE grid size, dispersion, pump power, time step |
| 0 | `zeta_end`, `HOLD_TAU` | how far the detuning scan goes and how long the steady state is held |
| 0 | `USABLE_BUDGET_DB` | how far down the envelope counts as "usable" |
| 1 | `PILOT_LEN`, `SYMBOL_RATE_HZ`, `SD_FEC_BER` | pilot overhead, baud rate, FEC threshold |
| 1 | `SCRAMBLE_MIN_AMP`, `SCRAMBLE_MAX_AMP` | the amplitude mask range (set them equal for phase-only) |
| 1 | `dphi`, `sub_len` | phase-search window and hence the tracking range |
| 4 | `HOP_DWELL_SYMBOLS`, `DUMMY_TRAFFIC`, `OSNR_REF_DB`, `OSNR_THRESHOLD_DB` | dwell length, dummy traffic on/off, reference OSNR, usable-line criterion |
| 5 | `OSNR_SWEEP_DB`, `N_SWEEP`, `SWEEP_SEEDS` | sweep range, symbols per point (the resolution floor is `3/N`), repetitions |
