pub fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    let (_, upper, _) = values.select_nth_unstable_by(n / 2, f64::total_cmp);
    let upper = *upper;
    if n % 2 == 1 {
        return upper;
    }
    let lower = values[..n / 2].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (lower + upper) / 2.0
}

pub fn robust_peak_z(profile: &[f64], scratch: &mut Vec<f64>) -> f64 {
    scratch.clear();
    scratch.extend_from_slice(profile);
    let med = median(scratch);
    let peak = profile.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    scratch.clear();
    scratch.extend(profile.iter().map(|v| (v - med).abs()));
    let mad = median(scratch);
    (peak - med) / (1.4826 * mad + 1e-9)
}

pub fn round_half_even(x: f64) -> i64 {
    x.round_ties_even() as i64
}

pub fn fisher_significance(half_statistic: f64, passes: usize) -> f64 {
    let ln_series = (1..passes)
        .scan(0.0, |ln_term, i| {
            *ln_term += half_statistic.ln() - (i as f64).ln();
            Some(*ln_term)
        })
        .fold(0.0, |acc: f64, t| acc.max(t) + (-(acc - t).abs()).exp().ln_1p());
    (half_statistic - ln_series) / std::f64::consts::LN_10
}
