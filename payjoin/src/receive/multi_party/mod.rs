use bitcoin::{FeeRate, Psbt};
use error::InternalMultiPartyError;

use super::error::InputContributionError;
use super::{v1, v2, Error, InputPair, SelectionError};
use crate::error::MultiPartyError;
use crate::psbt::PsbtExt;
use crate::receive::v2::SessionContext;

pub(crate) mod error;

const SUPPORTED_VERSIONS: &[usize] = &[2];

pub struct MultiPartyProposalBuilder {
    proposals: Vec<v2::UncheckedProposal>,
}

impl MultiPartyProposalBuilder {
    pub fn new() -> Self { Self { proposals: vec![] } }

    pub fn try_add(
        &mut self,
        proposal: v2::UncheckedProposal,
    ) -> Result<&mut Self, MultiPartyError> {
        self.check_proposal_suitability(&proposal)?;
        self.proposals.push(proposal);
        Ok(self)
    }

    fn check_proposal_suitability(
        &self,
        proposal: &v2::UncheckedProposal,
    ) -> Result<(), MultiPartyError> {
        let params = proposal.params();
        if !SUPPORTED_VERSIONS.contains(&params.v) {
            return Err(InternalMultiPartyError::ProposalVersionNotSupported(params.v).into());
        }

        if let Some(opt_merge) = params.optimistic_merge {
            if !opt_merge {
                return Err(InternalMultiPartyError::OptimisticMergeNotSupported.into());
            }
        } else {
            return Err(InternalMultiPartyError::OptimisticMergeNotSupported.into());
        }
        Ok(())
    }

    pub fn try_build(&self) -> Result<MultiPartyProposal, MultiPartyError> {
        if self.proposals.len() < 2 {
            return Err(InternalMultiPartyError::NotEnoughProposals.into());
        }

        let mut agg_psbt = self.proposals.get(0).expect("checked above").psbt().clone();
        agg_psbt.dangerous_clear_signatures();
        let agg_psbt = self
            .proposals
            .iter()
            .skip(1) // Skip first since we already have it as agg_psbt
            .map(|p| p.psbt())
            .try_fold(agg_psbt, |mut acc, other| {
                let mut other = other.clone();
                other.dangerous_clear_signatures();
                acc.merge_unsigned_tx(other)
                    .map_err(|e| InternalMultiPartyError::FailedToMergeProposals(e))?;
                Ok::<_, InternalMultiPartyError>(acc)
            })?;

        let unchecked_proposal = v1::UncheckedProposal {
            psbt: agg_psbt,
            params: self.proposals.get(0).expect("checked above").params().clone(),
        };
        let sender_contexts = self.proposals.iter().map(|p| p.context().clone()).collect();
        // TODO(armins) we are just using the params from the first proposal. Probably should validate and merge them somehow
        Ok(MultiPartyProposal { v1: unchecked_proposal, sender_contexts })
    }
}

/// A multi-party proposal that has been merged by the receiver
pub struct MultiPartyProposal {
    v1: v1::UncheckedProposal,
    sender_contexts: Vec<SessionContext>,
}

impl MultiPartyProposal {
    pub fn check_broadcast_suitability(
        self,
        min_fee_rate: Option<FeeRate>,
        can_broadcast: impl Fn(&bitcoin::Transaction) -> Result<bool, Error>,
    ) -> Result<MultiPartyPayjoinProposalMaybeInputsOwned, Error> {
        let inner = self.v1.check_broadcast_suitability(min_fee_rate, can_broadcast)?;
        Ok(MultiPartyPayjoinProposalMaybeInputsOwned {
            v1: inner,
            sender_contexts: self.sender_contexts,
        })
    }
}

pub struct MultiPartyPayjoinProposalMaybeInputsOwned {
    v1: v1::MaybeInputsOwned,
    sender_contexts: Vec<SessionContext>,
}

impl MultiPartyPayjoinProposalMaybeInputsOwned {
    pub fn check_inputs_not_owned(
        self,
        is_owned: impl Fn(&bitcoin::Script) -> Result<bool, Error>,
    ) -> Result<MultiPartyPayjoinProposalMaybeInputsSeen, Error> {
        let inner = self.v1.check_inputs_not_owned(is_owned)?;
        Ok(MultiPartyPayjoinProposalMaybeInputsSeen {
            v1: inner,
            sender_contexts: self.sender_contexts,
        })
    }
}

pub struct MultiPartyPayjoinProposalMaybeInputsSeen {
    v1: v1::MaybeInputsSeen,
    sender_contexts: Vec<SessionContext>,
}

impl MultiPartyPayjoinProposalMaybeInputsSeen {
    pub fn check_no_inputs_seen_before(
        self,
        is_seen: impl Fn(&bitcoin::OutPoint) -> Result<bool, Error>,
    ) -> Result<MultiPartyPayjoinProposalOutputsUnknown, Error> {
        let inner = self.v1.check_no_inputs_seen_before(is_seen)?;
        Ok(MultiPartyPayjoinProposalOutputsUnknown {
            v1: inner,
            sender_contexts: self.sender_contexts,
        })
    }
}

pub struct MultiPartyPayjoinProposalOutputsUnknown {
    v1: v1::OutputsUnknown,
    sender_contexts: Vec<SessionContext>,
}

impl MultiPartyPayjoinProposalOutputsUnknown {
    pub fn identify_receiver_outputs(
        self,
        is_receiver_output: impl Fn(&bitcoin::Script) -> Result<bool, Error>,
    ) -> Result<MultiPartyPayjoinProposalWantsOutputs, Error> {
        let inner = self.v1.identify_receiver_outputs(is_receiver_output)?;
        Ok(MultiPartyPayjoinProposalWantsOutputs {
            v1: inner,
            sender_contexts: self.sender_contexts,
        })
    }
}

pub struct MultiPartyPayjoinProposalWantsOutputs {
    v1: v1::WantsOutputs,
    sender_contexts: Vec<SessionContext>,
}

impl MultiPartyPayjoinProposalWantsOutputs {
    pub fn commit_outputs(self) -> MultiPartyPayjoinProposalWantsInputs {
        let inner = self.v1.commit_outputs();
        MultiPartyPayjoinProposalWantsInputs { v1: inner, sender_contexts: self.sender_contexts }
    }
}

pub struct MultiPartyPayjoinProposalWantsInputs {
    v1: v1::WantsInputs,
    sender_contexts: Vec<SessionContext>,
}

impl MultiPartyPayjoinProposalWantsInputs {
    pub fn try_preserving_privacy(
        &self,
        candidate_inputs: impl IntoIterator<Item = InputPair>,
    ) -> Result<InputPair, SelectionError> {
        self.v1.try_preserving_privacy(candidate_inputs)
    }

    /// Add the provided list of inputs to the transaction.
    /// Any excess input amount is added to the change_vout output indicated previously.
    pub fn contribute_inputs(
        self,
        inputs: impl IntoIterator<Item = InputPair>,
    ) -> Result<MultiPartyPayjoinProposalWantsInputs, InputContributionError> {
        let inner = self.v1.contribute_inputs(inputs)?;
        Ok(MultiPartyPayjoinProposalWantsInputs {
            v1: inner,
            sender_contexts: self.sender_contexts,
        })
    }

    /// Proceed to the proposal finalization step.
    /// Inputs cannot be modified after this function is called.
    pub fn commit_inputs(self) -> MultiPartyProvisionalProposal {
        let inner = self.v1.commit_inputs();
        MultiPartyProvisionalProposal { v1: inner, sender_contexts: self.sender_contexts }
    }
}

pub struct MultiPartyProvisionalProposal {
    v1: v1::ProvisionalProposal,
    sender_contexts: Vec<SessionContext>,
}

impl MultiPartyProvisionalProposal {
    pub fn finalize_proposal(
        self,
        wallet_process_psbt: impl Fn(&Psbt) -> Result<Psbt, Error>,
        min_feerate_sat_per_vb: Option<FeeRate>,
        max_feerate_sat_per_vb: FeeRate,
    ) -> Result<MultiPartyPayjoinProposal, Error> {
        let inner = self.v1.finalize_proposal(
            wallet_process_psbt,
            min_feerate_sat_per_vb,
            max_feerate_sat_per_vb,
        )?;
        Ok(MultiPartyPayjoinProposal { v1: inner, sender_contexts: self.sender_contexts })
    }
}

pub struct MultiPartyPayjoinProposal {
    v1: v1::PayjoinProposal,
    sender_contexts: Vec<SessionContext>,
}

impl MultiPartyPayjoinProposal {
    pub fn sender_iter(&self) -> impl Iterator<Item = v2::PayjoinProposal> {
        self.sender_contexts
            .iter()
            .map(|ctx| v2::PayjoinProposal::new(self.v1.clone(), ctx.clone()))
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn proposal(&self) -> &v1::PayjoinProposal { &self.v1 }
}
