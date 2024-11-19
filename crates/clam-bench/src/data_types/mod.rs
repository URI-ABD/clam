//! Data Types that are used in the benchmarks.

/// The data types that can be used for the input dataset.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum Dataset {
    /// 2d-array in the `.npy` format.
    #[clap(name = "array")]
    Array,
    /// Set data. A plain-text file in which each row is a space-separated set
    /// of elements enumerated as unsigned integers.
    #[clap(name = "sets")]
    Sets,
    /// Sequences in a FASTA file.
    #[clap(name = "fasta")]
    Fasta,
}

/// The data types that can be used.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum Units {
    /// f32.
    #[clap(name = "f32")]
    F32,
    /// f64.
    #[clap(name = "f64")]
    F64,
    /// i8.
    #[clap(name = "i8")]
    I8,
    /// i16.
    #[clap(name = "i16")]
    I16,
    /// i32.
    #[clap(name = "i32")]
    I32,
    /// i64.
    #[clap(name = "i64")]
    I64,
    /// isize.
    #[clap(name = "isize")]
    Isize,
    /// u8.
    #[clap(name = "u8")]
    U8,
    /// u16.
    #[clap(name = "u16")]
    U16,
    /// u32.
    #[clap(name = "u32")]
    U32,
    /// u64.
    #[clap(name = "u64")]
    U64,
    /// usize.
    #[clap(name = "usize")]
    Usize,
    /// bool.
    #[clap(name = "bool")]
    Bool,
    /// Char.
    #[clap(name = "char")]
    Char,
    /// Strings.
    #[clap(name = "string")]
    String,
}
