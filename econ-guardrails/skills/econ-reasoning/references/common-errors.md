# Common Economics Errors — Worked Examples

## Example 1: Sign Convention Inconsistency

**Wrong:**
> "OPEC typically offsets 30-70% of non-OPEC supply changes, with an offset ratio of 0.5.
> The coefficient on OPEC response is -0.5, meaning for every 1 mb/d increase in
> non-OPEC supply, OPEC reduces output by 0.5 mb/d."

**Problem:** The "offset ratio" of 0.5 is described positively (OPEC offsets 50%), but the coefficient is -0.5. The verbal framing says "offsets 30-70%" (positive proportion) while the math says -0.5 (negative). A reader cannot tell if the net effect is +0.5 or -0.5 of the original shock.

**Correct:**
> "OPEC typically offsets 30-70% of non-OPEC supply changes. Defining the offset
> coefficient as the reduction in OPEC output per unit increase in non-OPEC output,
> the coefficient is -0.5: for every +1 mb/d non-OPEC increase, OPEC cuts 0.5 mb/d.
> The net supply change is therefore +0.5 mb/d (= 1.0 - 0.5)."

**Fix:** Define the sign convention explicitly before using it. State the net effect.

---

## Example 2: Mechanism Confusion

**Wrong:**
> "In the long run, the oil market rebalances through demand substitution —
> consumers switch away from oil, which is why OPEC's long-run offset ratio
> is higher than the short-run ratio."

**Problem:** Long-run rebalancing via higher OPEC offset ratios is a *supply-side* mechanism (OPEC adjusting production quotas over time), not demand substitution. These are independent channels.

**Correct:**
> "In the long run, the oil market rebalances through two channels:
> (1) Supply-side: OPEC adjusts quotas, raising the offset ratio from ~0.3
> (short-run) to ~0.5-0.7 (long-run);
> (2) Demand-side: consumers substitute away from oil (long-run demand elasticity
> of -0.3 to -0.6 vs short-run -0.02 to -0.10)."

---

## Example 3: Missing Horizon

**Wrong:**
> "The price elasticity of oil demand is approximately -0.3."

**Problem:** -0.3 is a long-run estimate. Short-run elasticity is -0.02 to -0.10. Using -0.3 for a 1-year shock analysis overstates the demand response by 3-15x.

**Correct:**
> "The short-run price elasticity of oil demand (within 1 year) is approximately
> -0.05, while the long-run elasticity (5+ years) is approximately -0.3
> (Hamilton, 2009; Kilian & Murphy, 2014)."

---

## Example 4: Causal Direction Error

**Wrong:**
> "Since the Taylor rule shows that a 1pp increase in inflation leads to a
> 1.5pp increase in the federal funds rate, we can conclude that raising rates
> by 150bps would reduce inflation by 1pp."

**Problem:** The Taylor rule is a *reaction function* — it describes what the Fed does in response to inflation. You cannot invert it to claim that rate changes cause specific inflation outcomes. That requires a structural model of the monetary transmission mechanism.

**Correct:**
> "The Taylor rule describes the Fed's systematic response: historically,
> a 1pp inflation increase is associated with a ~1.5pp rate increase.
> To estimate the *effect* of rate changes on inflation, we need structural
> identification (e.g., Romer & Romer (2004) narrative shocks, or SVAR
> with sign restrictions). These estimates suggest a 100bps tightening
> reduces inflation by 0.2-0.5pp over 2-3 years."

---

## Example 5: Unit Mixing

**Wrong:**
> "Venezuela's production fell by 1.5 million barrels. At $70/barrel, that's
> $105 million in lost revenue."

**Problem:** 1.5 million barrels of *what*? Per day? Total? If it's mb/d, the annual revenue loss is $105M × 365 = $38.3 billion, not $105 million.

**Correct:**
> "Venezuela's production fell by 1.5 mb/d. At $70/barrel, the daily revenue
> loss is 1.5M × $70 = $105M/day, or approximately $38.3B/year."

---

## Example 6: Stock vs Flow Confusion

**Wrong:**
> "Venezuela has 300 billion barrels of oil reserves, so at current production
> rates of 0.8 mb/d, they can increase revenue by 300B × $70 = $21 trillion."

**Problem:** Reserves (stock) cannot be directly multiplied by price to get revenue. Revenue is price × flow (production rate). The reserves determine how long production can be sustained, not instantaneous revenue.

**Correct:**
> "Venezuela has 300 billion barrels of proved reserves. At 0.8 mb/d production,
> annual revenue is 0.8M × 365 × $70 ≈ $20.4B/year. The reserve-to-production
> ratio is 300B / (0.8M × 365) ≈ 1,027 years at current rates (though
> economically recoverable reserves are much smaller)."
