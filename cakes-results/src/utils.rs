//! Utility functions for the crate.

use num_format::ToFormattedString;

/// Format a `f32` as a string with 6 digits of precision and separators.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub fn format_f32(x: f32) -> String {
    let trunc = x.trunc() as u32;
    let fract = (x.fract() * 10f32.powi(3)).round() as u32;

    let trunc = trunc.to_formatted_string(&num_format::Locale::en);

    #[allow(clippy::unwrap_used)]
    let fract = fract.to_formatted_string(
        &num_format::CustomFormat::builder()
            .grouping(num_format::Grouping::Standard)
            .separator("_")
            .build()
            .unwrap(),
    );

    format!("{trunc}.{fract}")
}
