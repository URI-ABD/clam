use super::types::*;

#[derive(Debug, Eq, PartialEq)]
pub struct Dataset {
    pub data: Box<Data>,
    pub metric: String,
}
