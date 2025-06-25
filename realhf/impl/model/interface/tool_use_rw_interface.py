# Copyright 2025 Tool-Use AReAL Integration

import asyncio
import dataclasses
import json
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

import realhf.api.core.model_api as model_api
import realhf.base.logging as logging
from realhf.api.core.data_api import (
    MicroBatchSpec,
    SequenceSample,
    load_hf_tokenizer,
)
from realhf.base import constants
from realhf.base.datapack import flat2d

logger = logging.getLogger("Tool-Use Reward Interface", "benchmark")


def extract_answer_from_text(text: str) -> str:
    """
    Extract answer from tool-use response text.
    
    Looks for answer tool calls in the format:
    {"function": {"name": "answer", "arguments": {"answer": "VALUE"}}}
    """
    try:
        # Look for answer tool calls
        import re
        
        # Pattern for answer tool calls
        answer_pattern = r'"function":\s*{\s*"name":\s*"answer"[^}]*"arguments":\s*{\s*"answer":\s*"([^"]*)"'
        matches = re.findall(answer_pattern, text)
        
        if matches:
            return matches[-1].strip()  # Return last answer
            
        # Fallback: look for any JSON-like answer structure
        json_pattern = r'{"answer":\s*"([^"]*)"}'
        matches = re.findall(json_pattern, text)
        
        if matches:
            return matches[-1].strip()
            
        # Last fallback: return the text itself (for evaluation)
        return text.strip()
        
    except Exception as e:
        logger.warning(f"Failed to extract answer from text: {e}")
        return text.strip()


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""

    if not s or s.isspace():
        return ""

    import re
    import string

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    result = lower(s)
    result = remove_punc(result)
    result = remove_articles(result)
    result = white_space_fix(result)

    return result


def f1_score_cal(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score."""
    if prediction is None or ground_truth is None:
        return 0.0

    from collections import Counter
    
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def em_check(pred: str, answer: str) -> Tuple[int, float]:
    """Check both exact match and F1 score."""
    if pred is None or answer is None:
        return 0, 0.0

    normalized_pred = normalize_answer(pred)
    normalized_ans = normalize_answer(answer)

    # EM check
    if not normalized_pred and not normalized_ans:
        em_score = 1
    elif not normalized_pred or not normalized_ans:
        em_score = 0
    else:
        em_score = 1 if normalized_ans == normalized_pred else 0

    # F1 check
    f1_score = f1_score_cal(pred, answer)

    return em_score, f1_score


def validate_tool_call_format(text: str) -> bool:
    """
    Validate that text contains properly formatted tool calls.
    
    Args:
        text: Response text to validate
        
    Returns:
        True if response contains valid tool call format
    """
    try:
        import re
        
        # Look for tool call patterns
        tool_call_pattern = r'"function":\s*{\s*"name":\s*"[^"]*"[^}]*"arguments":\s*{[^}]*}'
        
        if re.search(tool_call_pattern, text):
            return True
            
        # Also accept simple JSON format
        json_pattern = r'{"[^"]*":\s*"[^"]*"}'
        if re.search(json_pattern, text):
            return True
            
        return False
        
    except Exception:
        return False


def load_ground_truth_metadata(dataset_path: str) -> Dict[str, str]:
    """
    Load ground truth answers from dataset file.
    
    Args:
        dataset_path: Path to JSONL dataset file
        
    Returns:
        Dictionary mapping query IDs to ground truth answers
    """
    id2answer = {}
    
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset file not found: {dataset_path}")
        return id2answer
        
    try:
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # Extract query ID and answer
                qid = str(data.get('id', data.get('query_id', '')))
                answer = data.get('answer', data.get('target', data.get('ground_truth', '')))
                
                if qid and answer:
                    id2answer[qid] = str(answer)
                    
    except Exception as e:
        logger.error(f"Failed to load ground truth from {dataset_path}: {e}")
        
    logger.info(f"Loaded {len(id2answer)} ground truth answers")
    return id2answer


def compute_tool_use_rewards(
    answers: List[str], 
    query_ids: List[str],
    id2answer: Dict[str, str],
    correctness_weight: float = 1.0,
    format_weight: float = 0.2,
    scoring_method: str = "f1"
) -> List[float]:
    """
    Compute rewards for tool-use responses.
    
    Args:
        answers: List of response texts
        query_ids: List of query IDs  
        id2answer: Ground truth mapping
        correctness_weight: Weight for correctness score
        format_weight: Weight for format validation
        scoring_method: "f1" or "em"
        
    Returns:
        List of scalar rewards per response
    """
    rewards = []
    
    for answer, qid in zip(answers, query_ids):
        reward = 0.0
        
        try:
            # Get ground truth
            ground_truth = id2answer.get(qid, "")
            
            # Extract answer from response
            extracted_answer = extract_answer_from_text(answer)
            
            # Compute correctness score
            correctness_score = 0.0
            if extracted_answer and ground_truth:
                em_score, f1_score = em_check(extracted_answer, ground_truth)
                correctness_score = f1_score if scoring_method == "f1" else float(em_score)
            
            # Compute format score
            format_score = 1.0 if validate_tool_call_format(answer) else 0.0
            
            # Combine scores
            reward = correctness_score * correctness_weight + format_score * format_weight
            
        except Exception as e:
            logger.warning(f"Error computing reward for query {qid}: {e}")
            reward = 0.0
            
        rewards.append(reward)
    
    return rewards


@dataclasses.dataclass
class ToolUseRewardInterface(model_api.ModelInterface):
    """
    AReAL-compatible reward interface for tool-use agent training.
    
    This interface computes rewards for tool-use responses based on:
    1. Correctness (EM/F1 scoring against ground truth)
    2. Format validation (proper tool call structure)
    """
    dataset_path: str = ""
    tokenizer_path: str = "/storage/models/Qwen__Qwen2.5-1.5B"
    output_scaling: float = 1.0
    output_bias: float = 0.0
    correctness_weight: float = 1.0
    format_weight: float = 0.2
    scoring_method: str = "f1"
    group_size: int = 1
    
    def __post_init__(self):
        # Load ground truth data
        self.id2answer = load_ground_truth_metadata(self.dataset_path)
        
        # Load tokenizer
        self.tokenizer = load_hf_tokenizer(self.tokenizer_path)
        
        if constants.parallelism_rank() == 0:
            logger.info(f"Tool-Use Reward Interface initialized")
            logger.info(f"  correctness_weight: {self.correctness_weight}")
            logger.info(f"  format_weight: {self.format_weight}")
            logger.info(f"  scoring_method: {self.scoring_method}")
            logger.info(f"  output_scaling: {self.output_scaling}")
            logger.info(f"  output_bias: {self.output_bias}")
            logger.info(f"  loaded {len(self.id2answer)} ground truth answers")

    def _retokenize_and_verify(
        self,
        prompt_ids: List[List[int]],
        seq_ids: List[List[int]], 
        query_ids: List[str]
    ) -> List[float]:
        """
        Retokenize sequences and compute rewards.
        
        Args:
            prompt_ids: List of prompt token sequences
            seq_ids: List of complete token sequences (prompt + response)
            query_ids: List of query IDs
            
        Returns:
            List of scalar rewards
        """
        # Decode sequences
        seq_strs = self.tokenizer.batch_decode(
            seq_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
        )
        prompt_strs = self.tokenizer.batch_decode(
            prompt_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
        )
        
        # Extract response portions
        answers = []
        for seq_str, prompt_str in zip(seq_strs, prompt_strs):
            try:
                # Get response by removing prompt
                response = seq_str.split(prompt_str, 1)[1] if prompt_str in seq_str else seq_str
                answers.append(response)
            except Exception as e:
                logger.warning(f"Failed to extract response: {e}")
                answers.append(seq_str)
        
        # Clean query IDs (remove @ suffixes if present)
        clean_query_ids = [qid.split("@")[0] for qid in query_ids]
        
        # Compute rewards
        rewards = compute_tool_use_rewards(
            answers=answers,
            query_ids=clean_query_ids,
            id2answer=self.id2answer,
            correctness_weight=self.correctness_weight,
            format_weight=self.format_weight,
            scoring_method=self.scoring_method
        )
        
        return rewards

    def _dispatch_tp_and_pp(self, data: SequenceSample):
        """Handle tensor/pipeline parallelism dispatch."""
        tp_pp_size = constants.tp_and_pp_world_size()
        if tp_pp_size == 1:
            return data, None
            
        splitted, _, backward_indices = data.split(
            mb_spec=MicroBatchSpec(n_mbs=tp_pp_size)
        )
        tp_pp_rank = constants.tp_and_pp_rank()
        return splitted[tp_pp_rank], backward_indices

    def _gather_tp_and_pp(self, input_, data: SequenceSample, backward_indices):
        """Handle tensor/pipeline parallelism gather."""
        tp_pp_size = constants.tp_and_pp_world_size()
        if tp_pp_size == 1:
            return data
            
        local_rank = constants.grid().topo.get_rank(
            data=constants.data_parallel_rank(),
            tensor=0,
            pipe=constants.pipe_parallel_world_size() - 1,
        )
        dst = constants.to_global_pg_rank(local_rank)
        gather_list = None
        if dist.get_rank() == dst:
            gather_list = [None for _ in range(tp_pp_size)]
            
        x = data.data["rewards"].cpu().numpy().tolist()
        dist.gather_object(
            x, gather_list, dst=dst, group=constants.tp_and_pp_cpu_group()
        )
        
        if dist.get_rank() != dst:
            return None
            
        import numpy as np
        gathered = np.array(gather_list).reshape(-1, self.group_size)
        assert len(gathered) == len(backward_indices)
        
        rewards = (
            np.concatenate([gathered[i] for i in backward_indices]).flatten().tolist()
        )
        
        return SequenceSample(
            keys=["rewards"],
            trailing_shapes=dict(rewards=()),
            dtypes=dict(rewards=torch.float32),
            ids=input_.ids,
            seqlens=dict(
                rewards=[[1 for _ in range(self.group_size)] for _ in range(input_.bs)],
            ),
            data=dict(rewards=torch.tensor(rewards, dtype=torch.float32)),
        )

    def calculate_tool_use_reward(
        self,
        model: model_api.Model,
        data: SequenceSample,
        mb_spec: MicroBatchSpec
    ):
        """
        Calculate rewards for tool-use responses.
        
        Args:
            model: Model instance (unused for reward computation)
            data: SequenceSample with packed sequences
            mb_spec: Micro-batch specification
            
        Returns:
            SequenceSample with rewards
        """
        # Extract token sequences
        packed_input_ids: torch.Tensor = data.data["packed_input_ids"]
        input_seqlens = flat2d(data.seqlens["packed_input_ids"])
        
        # Unpack sequences
        seq_ids = []
        offset = 0
        for slen in input_seqlens:
            seq_ids.append(
                packed_input_ids[offset : offset + slen].cpu().numpy().tolist()
            )
            offset += slen
        assert offset == packed_input_ids.shape[0]
        
        # Extract prompts
        prompt_input_ids = data.data["packed_prompts"]
        prompt_len = flat2d(data.seqlens["packed_prompts"])
        prompt_ids = []
        offset = 0
        for slen in prompt_len:
            p = prompt_input_ids[offset : offset + slen].cpu().numpy().tolist()
            prompt_ids += [p] * self.group_size  # Repeat for group_size
            offset += slen
            
        # Get query IDs
        query_ids = [
            str(data_id) for data_id in data.ids for _ in range(self.group_size)
        ]
        
        # Compute rewards
        raw_rewards = self._retokenize_and_verify(
            prompt_ids=prompt_ids,
            seq_ids=seq_ids,
            query_ids=query_ids
        )
        
        # Apply scaling and bias
        scores = torch.FloatTensor(raw_rewards).to(packed_input_ids.device)
        scores = (scores - self.output_bias) * self.output_scaling
        
        # Log results
        self._log_rewards(model, raw_rewards, scores)
        
        # Create result SequenceSample
        res = SequenceSample(
            keys=["rewards"],
            trailing_shapes=dict(rewards=()),
            dtypes=dict(rewards=torch.float32),
            ids=data.ids,
            seqlens=dict(
                rewards=[
                    [1 for _ in range(len(x))] for x in data.seqlens["packed_input_ids"]
                ],
            ),
            data=dict(rewards=scores),
        )
        
        # Add metadata
        avg_scores = []
        offset = 0
        for i in range(data.bs):
            score_list = scores[
                offset : offset + len(data.seqlens["packed_input_ids"][i])
            ]
            avg_scores.append(score_list.mean().item())
            offset += len(data.seqlens["packed_input_ids"][i])
            
        res.metadata["scores"] = avg_scores
        
        return res

    def _log_rewards(self, model: model_api.Model, raw_rewards: List[float], scaled_scores: torch.Tensor):
        """Log reward computation results."""
        if constants.parallelism_rank() == 0:
            avg_raw = sum(raw_rewards) / len(raw_rewards) if raw_rewards else 0.0
            avg_scaled = scaled_scores.mean().item() if len(scaled_scores) > 0 else 0.0
            
            logger.info(f"Tool-use rewards computed: {len(raw_rewards)} samples")
            logger.info(f"  avg raw reward: {avg_raw:.4f}")
            logger.info(f"  avg scaled reward: {avg_scaled:.4f}")
            logger.info(f"  max reward: {max(raw_rewards) if raw_rewards else 0.0:.4f}")
            logger.info(f"  min reward: {min(raw_rewards) if raw_rewards else 0.0:.4f}")

    def inference(
        self,
        model: model_api.Model,
        data: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample | None:
        """
        Main inference method for reward computation.
        
        Args:
            model: Model instance
            data: Input SequenceSample with sequences to evaluate
            mb_spec: Micro-batch specification
            
        Returns:
            SequenceSample with computed rewards
        """
        input_ = data
        data, backward_indices = self._dispatch_tp_and_pp(data)
        
        # Compute rewards
        result = self.calculate_tool_use_reward(model, data, mb_spec)
        
        # Handle distributed gather
        final_result = self._gather_tp_and_pp(input_, result, backward_indices)
        
        # Update model version
        model.inc_version()
        
        return final_result


# Register the interface with AReAL
model_api.register_interface("rw-tool-use", ToolUseRewardInterface)