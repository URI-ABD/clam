//! The modulation modes in the `RadioML` dataset.

/// The modulation modes in the `RadioML` dataset.
#[allow(non_camel_case_types, missing_docs)]
pub enum ModulationMode {
    APSK_128,
    APSK_64,
    APSK_32,
    APSK_16,
    PSK_32,
    PSK_16,
    PSK_8,
    ASK_8,
    ASK_4,
    QAM_256,
    QAM_128,
    QAM_64,
    QAM_32,
    QAM_16,
    AM_DSB_SC,
    AM_DSB_WC,
    AM_SSB_SC,
    AM_SSB_WC,
    BPSK,
    FM,
    GMSK,
    OOK,
    OQPSK,
    QPSK,
}

impl ModulationMode {
    /// Returns all the modulation modes in the `RadioML` dataset.
    #[must_use]
    pub const fn all() -> [Self; 24] {
        [
            Self::APSK_128,
            Self::APSK_64,
            Self::APSK_32,
            Self::APSK_16,
            Self::PSK_32,
            Self::PSK_16,
            Self::PSK_8,
            Self::ASK_8,
            Self::ASK_4,
            Self::QAM_256,
            Self::QAM_128,
            Self::QAM_64,
            Self::QAM_32,
            Self::QAM_16,
            Self::AM_DSB_SC,
            Self::AM_DSB_WC,
            Self::AM_SSB_SC,
            Self::AM_SSB_WC,
            Self::BPSK,
            Self::FM,
            Self::GMSK,
            Self::OOK,
            Self::OQPSK,
            Self::QPSK,
        ]
    }

    /// Returns the name of the modulation mode.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::APSK_128 => "128APSK",
            Self::APSK_64 => "64APSK",
            Self::APSK_32 => "32APSK",
            Self::APSK_16 => "16APSK",
            Self::PSK_32 => "32PSK",
            Self::PSK_16 => "16PSK",
            Self::PSK_8 => "8PSK",
            Self::ASK_8 => "8ASK",
            Self::ASK_4 => "4ASK",
            Self::QAM_256 => "256QAM",
            Self::QAM_128 => "128QAM",
            Self::QAM_64 => "64QAM",
            Self::QAM_32 => "32QAM",
            Self::QAM_16 => "16QAM",
            Self::AM_DSB_SC => "AM-DSB-SC",
            Self::AM_DSB_WC => "AM-DSB-WC",
            Self::AM_SSB_SC => "AM-SSB-SC",
            Self::AM_SSB_WC => "AM-SSB-WC",
            Self::BPSK => "BPSK",
            Self::FM => "FM",
            Self::GMSK => "GMSK",
            Self::OOK => "OOK",
            Self::OQPSK => "OQPSK",
            Self::QPSK => "QPSK",
        }
    }

    /// Returns the path to the `h5` file containing the modulation mode.
    pub fn h5_path<P: AsRef<std::path::Path>>(&self, inp_dir: &P) -> std::path::PathBuf {
        let mut path = inp_dir.as_ref().to_path_buf();
        path.push(format!("mod_{}.h5", self.name()));
        path
    }
}
