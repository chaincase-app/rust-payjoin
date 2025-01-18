use core::fmt;
use std::error;

use crate::psbt;
use crate::receive::v1::RequestError;

#[derive(Debug)]
pub struct MultiPartyError(InternalMultiPartyError);

#[derive(Debug)]
pub(crate) enum InternalMultiPartyError {
    /// Failed to merge proposals
    FailedToMergeProposals(Vec<psbt::MergePsbtError>),

    /// Bad Request
    BadRequest(RequestError),

    /// Not enough proposals
    NotEnoughProposals,

    /// Proposal version not supported
    ProposalVersionNotSupported(usize),

    /// Optimistic merge not supported
    OptimisticMergeNotSupported,
}

impl From<InternalMultiPartyError> for MultiPartyError {
    fn from(e: InternalMultiPartyError) -> Self { MultiPartyError(e) }
}

impl fmt::Display for MultiPartyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0 {
            InternalMultiPartyError::FailedToMergeProposals(e) =>
                write!(f, "Failed to merge proposals: {:?}", e),
            InternalMultiPartyError::BadRequest(e) => write!(f, "Bad Request: {}", e),
            InternalMultiPartyError::NotEnoughProposals => write!(f, "Not enough proposals"),
            InternalMultiPartyError::ProposalVersionNotSupported(v) =>
                write!(f, "Proposal version not supported: {}", v),
            InternalMultiPartyError::OptimisticMergeNotSupported =>
                write!(f, "Optimistic merge not supported"),
        }
    }
}

impl error::Error for MultiPartyError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self.0 {
            InternalMultiPartyError::FailedToMergeProposals(_) => None, // Vec<MergePsbtError> doesn't implement Error
            InternalMultiPartyError::BadRequest(e) => Some(e),
            InternalMultiPartyError::NotEnoughProposals => None,
            InternalMultiPartyError::ProposalVersionNotSupported(_) => None,
            InternalMultiPartyError::OptimisticMergeNotSupported => None,
        }
    }
}
