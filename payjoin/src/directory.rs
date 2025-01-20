#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortId(pub [u8; 8]);

impl ShortId {
    pub fn as_bytes(&self) -> &[u8] { &self.0 }
    pub fn as_slice(&self) -> &[u8] { &self.0 }
}

impl std::fmt::Display for ShortId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let id_hrp = bitcoin::bech32::Hrp::parse("ID").unwrap();
        f.write_str(
            crate::bech32::nochecksum::encode(id_hrp, &self.0)
                .expect("bech32 encoding of short ID must succeed")
                .strip_prefix("ID1")
                .expect("human readable part must be ID1"),
        )
    }
}

#[derive(Debug)]
pub enum ShortIdError {
    DecodeBech32(bitcoin::bech32::primitives::decode::CheckedHrpstringError),
    IncorrectLength(std::array::TryFromSliceError),
}

impl std::convert::From<bitcoin::hashes::sha256::Hash> for ShortId {
    fn from(h: bitcoin::hashes::sha256::Hash) -> Self {
        bitcoin::hashes::Hash::as_byte_array(&h)[..8]
            .try_into()
            .expect("truncating SHA256 to 8 bytes should always succeed")
    }
}

impl std::convert::TryFrom<&[u8]> for ShortId {
    type Error = ShortIdError;
    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let bytes: [u8; 8] = bytes.try_into().map_err(ShortIdError::IncorrectLength)?;
        Ok(Self(bytes))
    }
}

impl std::str::FromStr for ShortId {
    type Err = ShortIdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (_, bytes) = crate::bech32::nochecksum::decode(&("ID1".to_string() + s))
            .map_err(ShortIdError::DecodeBech32)?;
        (&bytes[..]).try_into()
    }
}
