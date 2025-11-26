import os
import gc

import itertools
import re
import json
import argparse
import pandas as pd
import torch

from llm_steer import LLM_Steer_Manager

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--itv-t", type=int, default=None)

    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=int(25))
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.5)
    
    parser.add_argument("--agent-a-id", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--agent-b-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    
    parser.add_argument("--verbose", type=bool, default=False)
    return parser.parse_args()


def extract_and_parse_json(text: str):
    if text is None:
        return {"error": "Input text is None"}

    # Try all minimally-matched blocks that look like JSON
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue

    return {"error": "NoValidJSONObjectParsed", "original_text": text}


def get_strategy_list(n):
    a_strategies = ['threat', 'deceive', 'cooperate', 'self-blame', 'give up']
    full = (a_strategies * (n // len(a_strategies))) + a_strategies[: (n % len(a_strategies))]
    return full


class Simulation_Config:
    def __init__(self, args, cfg, dm, model):
        self.verbose = args.verbose
        self.dm = dm
        self.symp_label_dict  = dm.load_dict(dict_type='label')[0] 
        self.device_dict = dm.load_dict(dict_type='device', model=model)

        self.layer_lambda_dict = dm.load_dict(dict_type='itv-str')
        self.sae_dict = dm.load_dict(dict_type='sae')
        for layer in cfg.hook_layers:
            self.sae_dict[layer] = self.sae_dict[layer].to(self.device_dict[layer])

        if args.itv_t is not None:
            self.itv_t = list(self.symp_label_dict.keys())[args.itv_t]
        else:
            self.itv_t = None

        self.generation_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'tmp': args.temperature,
        }
        self.NUM_ROUNDS = args.num_rounds
        self.NUM_SAMPLES = args.num_samples

        self.ALLOWED_ACTIONS = ['invest', 'no action', 'raid']
        self.ALLOWED_LEVELS = ['very low', 'low', 'moderate', 'high', 'very high']
        self.ALLOWED_STRATEGIES = ['threat', 'deceive', 'cooperate', 'self-blame', 'give up',]

        self.NOTE = (
            "You MUST ONLY generate a single, valid JSON object without any additional text, commentary, or markdown formatting."
        )

        # Prompts
        self.PROMPTS_A = {
            'strategy_selection': (
                "You are Agent-A in the game 'The Island Trader'.\n"
                "[Game rule]. {rule}\n"
                "[Game progress]."
                "\n- Current round: {round}/{total_rounds}."
                "\n- Your remaining Coconuts: {a_res}"
                "\n- Agent-B's remaining Coconuts: {b_res}"
                "\n- Conversation history from the previous rounds: {history}\n"
                "[Current task]. Choose your strategy for this round.\n"
                "[Instruction]."
                "\n- Provide ONLY the following JSON:\n{{\n  \"strategy\": \"[threat / deceive / cooperate / self-blame / give up]\"\n}}\n"
                "\n- This decision is PRIVATE and not shared with Agent-B."
                "\n- Choose exactly one from the provided options." 
                "\n- {note}"
            ),
            'message': (
                "You are Agent-A in the game 'The Island Trader'.\n"
                "[Game rule]. {rule}\n"
                "[Game progress]."
                "\n- Current round: {round}/{total_rounds}."
                "\n- Your remaining Coconuts: {a_res}"
                "\n- Agent-B's remaining Coconuts: {b_res}"
                "\n- Conversation history from the previous rounds: {history}\n"
                "\n- Your current strategy: {strategy}.\n"
                "[Current task]. Send a single short message to Agent-B.\n"
                "[Instruction]."
                "\n- Provide ONLY the following JSON:\n{{\n  \"Agent-A message\": \"<your message>\"\n}}\n"
                "\n- The message should be 2-4 sentences long."
                "\n- {note}"
            ),
            'action': (
                "You are Agent-A in the game 'The Island Trader'.\n"
                "[Game rule]. {rule}\n"
                "[Game progress]."
                "\n- Current round: {round}/{total_rounds}."
                "\n- Your remaining Coconuts: {a_res}"
                "\n- Agent-B's remaining Coconuts: {b_res}"
                "\n- Conversation history from the previous rounds: {history}\n"
                "\n- Your current strategy: {strategy}.\n"
                "\n- Your last message to Agent-B: {a_msg}\n"
                "\n- Agent-B's last reply message to you: {b_msg}\n"
                "[Current task]. Choose your action for this round.\n"
                "[Instruction]."
                "\n- Provide ONLY the following JSON:\n{{\n  \"Agent-A action\": \"[invest / no action / raid]\"\n}}\n"
                "\n- Choose exactly one from the provided options." 
                "\n- {note}"
            ),
        }

        self.PROMPTS_B = {
            'trust_evaluation': (
                "You are Agent-B in the game 'The Island Trader'.\n"
                "[Game rule]. {rule}\n"
                "[Game progress]."
                "\n- Current round: {round}/{total_rounds}."
                "\n- Your remaining Coconuts: {b_res}"
                "\n- Agent-A's remaining Coconuts: {a_res}"
                "\n- Conversation history from the previous rounds: {history}"
                "\n- Agent-A's last message: {a_msg}\n"
                "[Current task]. Evaluate your current trust in Agent-A.\n"
                "[Instruction]."
                "\n- Provide ONLY the following JSON:\n{{\n  \"trust_evaluation\": \"[very low / low / moderate / high / very high]\"\n}}\n"
                "\n- This analysis is PRIVATE and not shared with Agent-A."
                "\n- Choose exactly one from the provided options." 
                "\n- {note}"
            ),
            'strategy_selection': (
                "You are Agent-B in the game 'The Island Trader'.\n"
                "[Game rule]. {rule}\n"
                "[Game progress]."
                "\n- Current round: {round}/{total_rounds}."
                "\n- Your remaining Coconuts: {b_res}"
                "\n- Agent-A's remaining Coconuts: {a_res}"
                "\n- Conversation history from the previous rounds: {history}"
                "\n- Agent-A's last message: {a_msg}"
                "\n- Your current trust in Agent-A: {trust}.\n"
                "[Current task]. Choose your strategy for this round.\n"
                "[Instruction]."
                "\n- Provide ONLY the following JSON:\n{{\n  \"strategy\": \"[threat / deceive / cooperate / self-blame / give up]\"\n}}\n"
                "\n- This decision is PRIVATE and not shared with Agent-A."
                "\n- Choose exactly one from the provided options." 
                "\n- {note}"
            ),
            'message': (
                "You are Agent-B in the game 'The Island Trader'.\n"
                "[Game rule]. {rule}\n"
                "[Game progress]."
                "\n- Current round: {round}/{total_rounds}."
                "\n- Your remaining Coconuts: {b_res}"
                "\n- Agent-A's remaining Coconuts: {a_res}"
                "\n- Conversation history from the previous rounds: {history}"
                "\n- Agent-A's last message: {a_msg}"
                "\n- Your current trust in Agent-A: {trust}."
                "\n- Your current strategy: {strategy}.\n"
                "[Current task]. Send a single short message to Agent-A.\n"
                "[Instruction]."
                "\n- Provide ONLY the following JSON:\n{{\n  \"Agent-B message\": \"<your message>\"\n}}\n"
                "\n- The message should be 2-4 sentences long."
                "\n- {note}"
            ),
            'action': (
                "You are Agent-B in the game 'The Island Trader'.\n"
                "[Game rule]. {rule}\n"
                "[Game progress]."
                "\n- Current round: {round}/{total_rounds}."
                "\n- Your remaining Coconuts: {b_res}"
                "\n- Agent-A's remaining Coconuts: {a_res}"
                "\n- Conversation history from the previous rounds: {history}"
                "\n- Agent-A's last message: {a_msg}"
                "\n- Your current trust in Agent-A: {trust}."
                "\n- Your current strategy: {strategy}."
                "\n- Your reply message to Agent-A: {b_msg}\n"
                "[Current task]. Choose your action for this round.\n"
                "[Instruction]."
                "\n- Provide ONLY the following JSON:\n{{\n  \"Agent-B action\": \"[invest / no action / raid]\"\n}}\n"
                "\n- Choose exactly one from the provided options." 
                "\n- {note}"
            ),
        }
        
        self.RULE = (
            """
            The game involves two players, Agent-A and Agent-B. At the beginning of each round, both agents receive 2 Coconuts.\n\n
            Each round is divided into 3 phases:\n
            - Phase-1) Resource distribution: Both agents receive two Coconuts.\n
            - Phase-2) Communication: Agent-A sends a message to Agent-B, then Agent-B replies\n
            - Phase-3) Actions: Both agents simultaneously choose one action from [invest, no action, raid].\n\n
            Action outcomes:\n
            - Outcome-1) Both invest -> Pot doubles to 4, split equally -> +1 each\n
            - Outcome-2) Both no action -> No change -> 0 each\n
            - Outcome-3) Both raid -> Both lose all -> -2 each\n
            - Outcome-4) One invest, other no action -> investor -1, no action 0\n
            - Outcome-5) One invest, other Raids -> investor -2, raider +1\n
            - Outcome-6) One no action, other Raids -> no action -1, raider +1\n\n
            Game objective:\n
            - Maximize total Coconuts by the end of the game.\n
            """
        )


class AgentA:
    def __init__(self, cfg, simul_cfg, model, tokenizer):
        self.cfg = cfg
        self.simul_cfg = simul_cfg
        self.tokenizer = tokenizer
        self.llm_generator = LLM_Steer_Manager(cfg, model, tokenizer, self.simul_cfg.device_dict, self.simul_cfg.generation_kwargs)

    def _batch_decode(self, output_tokens):
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    def generate_messages(self, round_idx, history_batch, a_res_batch, b_res_batch, a_strategy_batch, max_retries=3):
        batch_size = len(history_batch)
        final = [None] * batch_size
        indices = list(range(batch_size))

        for attempt in range(max_retries):
            if not indices:
                break
            chats = []
            for i in range(len(indices)):
                j = indices[i]
                prompt = self.simul_cfg.PROMPTS_A['message'].format(
                    rule=self.simul_cfg.RULE,
                    strategy=a_strategy_batch[j],
                    round=round_idx + 1,
                    total_rounds=self.simul_cfg.NUM_ROUNDS,
                    a_res=a_res_batch[j],
                    b_res=b_res_batch[j],
                    history=history_batch[j],
                    note=self.simul_cfg.NOTE,
                )
                chats.append([{ 'role': 'user', 'content': prompt }])
            outputs = self.llm_generator.generate_text(chats)
            texts = self._batch_decode(outputs)
            failed = []
            for i, t in enumerate(texts):
                j = indices[i]
                parsed = extract_and_parse_json(t)
                if 'error' in parsed or ('Agent-A message' not in parsed):
                    failed.append(j)
                else:
                    # final[j] = parsed['Agent-A message']
                    final[j] = parsed.get('Agent-A message', None)
            indices = failed
        for i in range(batch_size):
            if final[i] is None:
                final[i] = "ERROR: Failed to generate Agent-A message."
        return final

    def generate_strategy(self, round_idx, history_batch, a_res_batch, b_res_batch, max_retries=3):
        batch_size = len(history_batch)
        a_strategies = [None] * batch_size

        indices = list(range(batch_size))
        for attempt in range(max_retries):
            if not indices:
                break
            chats = []
            for i in range(len(indices)):
                j = indices[i]
                prompt = self.simul_cfg.PROMPTS_A['strategy_selection'].format(
                    rule=self.simul_cfg.RULE,
                    round=round_idx + 1,
                    total_rounds=self.simul_cfg.NUM_ROUNDS,
                    a_res=a_res_batch[j],
                    b_res=b_res_batch[j],
                    history=history_batch[j],
                    note=self.simul_cfg.NOTE,
                )
                chats.append([{ 'role': 'user', 'content': prompt }])
                
            outputs = self.llm_generator.generate_text(chats)
            texts = self._batch_decode(outputs)
            failed = []
            for i, t in enumerate(texts):
                j = indices[i]
                parsed = extract_and_parse_json(t)
                v = parsed.get('strategy') if isinstance(parsed, dict) else None
                v = re.sub(r'[^\w\s]', '', v.lower().strip()) if isinstance(v, str) else v 
                if (v == 'selfblame') or (v == 'self blame'):
                    v = 'self-blame'
                if v == 'deception':
                    v = 'deceive'
                if v == 'cooperation':
                    v = 'cooperate'
                if v == 'giveup':
                    v = 'give up'
                if (v is None) or (v not in self.simul_cfg.ALLOWED_STRATEGIES):
                    failed.append(j)
                else:
                    a_strategies[j] = v
            indices = failed
        for i in range(batch_size):
            if a_strategies[i] is None:
                a_strategies[i] = 'ERROR: Failed to generate strategy for Agent-A.'

        return a_strategies
    
    def generate_actions(self, round_idx, history_batch, a_res_batch, b_res_batch, a_strategy_batch, a_msg_batch, b_msg_batch, max_retries=3):
        batch_size = len(history_batch)
        final = [None] * batch_size
        indices = list(range(batch_size))

        for attempt in range(max_retries):
            if not indices:
                break
            chats = []
            for i in range(len(indices)):
                j = indices[i]
                prompt = self.simul_cfg.PROMPTS_A['action'].format(
                    rule=self.simul_cfg.RULE,
                    strategy=a_strategy_batch[j],
                    round=round_idx + 1,
                    total_rounds=self.simul_cfg.NUM_ROUNDS,
                    a_res=a_res_batch[j],
                    b_res=b_res_batch[j],
                    history=history_batch[j],
                    a_msg=a_msg_batch[j],
                    b_msg=b_msg_batch[j],
                    note=self.simul_cfg.NOTE,
                )
                chats.append([{ 'role': 'user', 'content': prompt }])
            outputs = self.llm_generator.generate_text(chats)
            texts = self._batch_decode(outputs)
            failed = []
            for i, t in enumerate(texts):
                j = indices[i]
                parsed = extract_and_parse_json(t)
                v = parsed.get('Agent-A action') if isinstance(parsed, dict) else None
                if (v is None) or (v not in self.simul_cfg.ALLOWED_ACTIONS):
                    failed.append(j)
                else:
                    final[j] = v
                    
            indices = failed
        for i in range(batch_size):
            if final[i] is None:
                final[i] = 'no action'
        return final


class AgentB:
    def __init__(self, cfg, simul_cfg, model, tokenizer):            
        self.cfg = cfg
        self.simul_cfg = simul_cfg
        self.tokenizer = tokenizer
        self.llm_generator = LLM_Steer_Manager(cfg, model, tokenizer, self.simul_cfg.device_dict, self.simul_cfg.generation_kwargs)

    def _batch_decode(self, output_tokens):
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    def generate_analysis(self, itv_thought, round_idx, history_batch, a_res_batch, b_res_batch, a_msg_batch, max_retries=3):
        batch_size = len(history_batch)
        b_trusts = [None] * batch_size

        # Step 1: trust_evaluation
        indices = list(range(batch_size))
        for attempt in range(max_retries):
            if not indices:
                break
            chats = []
            for i in range(len(indices)):
                j = indices[i]
                prompt = self.simul_cfg.PROMPTS_B['trust_evaluation'].format(
                    rule=self.simul_cfg.RULE,
                    round=round_idx + 1,
                    total_rounds=self.simul_cfg.NUM_ROUNDS,
                    a_res=a_res_batch[j],
                    b_res=b_res_batch[j],
                    history=history_batch[j],
                    a_msg=a_msg_batch[j],
                    note=self.simul_cfg.NOTE,
                )
                chats.append([{ 'role': 'user', 'content': prompt }])
            if itv_thought:
                itv_W_dict, itv_str_dict = dict(), dict()
                for layer in self.cfg.hook_layers:
                    itv_W = self.simul_cfg.sae_dict[layer].decoder.weight.T[self.simul_cfg.symp_label_dict[itv_thought]]
                    itv_W_batch = itv_W.repeat(len(chats), 1, 1)
                    itv_W_dict[layer] = itv_W_batch
                    
                    itv_str = torch.tensor(self.simul_cfg.layer_lambda_dict[itv_thought][layer]).to(self.simul_cfg.device_dict[layer])
                    itv_str_batch = itv_str.repeat(len(chats), 1)
                    itv_str_dict[layer] = itv_str_batch
                outputs = self.llm_generator.generate_text_w_itv(chats, itv_W_dict, itv_str_dict)
            else:
                outputs = self.llm_generator.generate_text(chats)
            texts = self._batch_decode(outputs)
            failed = []
            for i, t in enumerate(texts):
                j = indices[i]
                parsed = extract_and_parse_json(t)
                v = parsed.get('trust_evaluation') if isinstance(parsed, dict) else None
                v = re.sub(r'[^\w\s]', '', v.lower().strip()) if isinstance(v, str) else v
                if (v is None) or (v not in self.simul_cfg.ALLOWED_LEVELS):
                    failed.append(j)
                else:
                    b_trusts[j] = v
            indices = failed
        for i in range(batch_size):
            if b_trusts[i] is None:
                b_trusts[i] = 'ERROR: Failed to generate trust evaluation.'

        return b_trusts

    def generate_strategy(self, itv_thought, round_idx, history_batch, a_res_batch, b_res_batch, a_msg_batch, b_trust_batch, max_retries=3):
        batch_size = len(history_batch)
        b_strategies = [None] * batch_size

        # Step 2: strategy_selection
        indices = list(range(batch_size))
        for attempt in range(max_retries):
            if not indices:
                break
            chats = []
            for i in range(len(indices)):
                j = indices[i]
                prompt = self.simul_cfg.PROMPTS_B['strategy_selection'].format(
                    rule=self.simul_cfg.RULE,
                    round=round_idx + 1,
                    total_rounds=self.simul_cfg.NUM_ROUNDS,
                    a_res=a_res_batch[j],
                    b_res=b_res_batch[j],
                    history=history_batch[j],
                    a_msg=a_msg_batch[j],
                    trust=b_trust_batch[j],
                    note=self.simul_cfg.NOTE,
                )
                chats.append([{ 'role': 'user', 'content': prompt }])
                
            if itv_thought:
                itv_W_dict, itv_str_dict = dict(), dict()
                for layer in self.cfg.hook_layers:
                    itv_W = self.simul_cfg.sae_dict[layer].decoder.weight.T[self.simul_cfg.symp_label_dict[itv_thought]]
                    itv_W_batch = itv_W.repeat(len(chats), 1, 1)
                    itv_W_dict[layer] = itv_W_batch
                    
                    itv_str = torch.tensor(self.simul_cfg.layer_lambda_dict[itv_thought][layer]).to(self.simul_cfg.device_dict[layer])
                    itv_str_batch = itv_str.repeat(len(chats), 1)
                    itv_str_dict[layer] = itv_str_batch
                outputs = self.llm_generator.generate_text_w_itv(chats, itv_W_dict, itv_str_dict)
            else:
                outputs = self.llm_generator.generate_text(chats)
                
            texts = self._batch_decode(outputs)
            failed = []
            for i, t in enumerate(texts):
                j = indices[i]
                parsed = extract_and_parse_json(t)
                v = parsed.get('strategy') if isinstance(parsed, dict) else None
                v = re.sub(r'[^\w\s]', '', v.lower().strip()) if isinstance(v, str) else v
                if (v == 'selfblame') or (v == 'self blame'):
                    v = 'self-blame'
                if v == 'deception':
                    v = 'deceive'
                if v == 'cooperation':
                    v = 'cooperate'
                if v == 'giveup':
                    v = 'give up'
                if (v is None) or (v not in self.simul_cfg.ALLOWED_STRATEGIES):
                    failed.append(j)
                else:
                    b_strategies[j] = v
            indices = failed
        for i in range(batch_size):
            if b_strategies[i] is None:
                b_strategies[i] = 'ERROR: Failed to generate strategy for Agent-B.'

        return b_strategies
    
    def generate_actions(self, itv_thought, round_idx, history_batch, a_res_batch, b_res_batch, b_trust_batch, b_strategy_batch, a_msg_batch, b_msg_batch, max_retries=3):
        batch_size = len(history_batch)
        final = [None] * batch_size
        
        # Step 3: action_generation
        indices = list(range(batch_size))
        for attempt in range(max_retries):
            if not indices:
                break
            chats = []
            for i in range(len(indices)):
                j = indices[i]
                prompt = self.simul_cfg.PROMPTS_B['action'].format(
                    rule=self.simul_cfg.RULE,
                    round=round_idx + 1,
                    total_rounds=self.simul_cfg.NUM_ROUNDS,
                    a_res=a_res_batch[j],
                    b_res=b_res_batch[j],
                    history=history_batch[j],
                    a_msg=a_msg_batch[j],
                    trust=b_trust_batch[j],
                    strategy=b_strategy_batch[j],
                    b_msg=b_msg_batch[j],
                    note=self.simul_cfg.NOTE,
                )
                chats.append([{ 'role': 'user', 'content': prompt }])
                
            if itv_thought:
                itv_W_dict, itv_str_dict = dict(), dict()
                for layer in self.cfg.hook_layers:
                    itv_W = self.simul_cfg.sae_dict[layer].decoder.weight.T[self.simul_cfg.symp_label_dict[itv_thought]]
                    itv_W_batch = itv_W.repeat(len(chats), 1, 1)
                    itv_W_dict[layer] = itv_W_batch
                    
                    itv_str = torch.tensor(self.simul_cfg.layer_lambda_dict[itv_thought][layer]).to(self.simul_cfg.device_dict[layer])
                    itv_str_batch = itv_str.repeat(len(chats), 1)
                    itv_str_dict[layer] = itv_str_batch
                outputs = self.llm_generator.generate_text_w_itv(chats, itv_W_dict, itv_str_dict)
            else:
                outputs = self.llm_generator.generate_text(chats)
                
            texts = self._batch_decode(outputs)
            failed = []
            for i, t in enumerate(texts):
                j = indices[i]
                parsed = extract_and_parse_json(t)
                v = parsed.get('Agent-B action') if isinstance(parsed, dict) else None
                v = re.sub(r'[^\w\s]', '', v.lower().strip()) if isinstance(v, str) else v
                if (v is None) or (v not in self.simul_cfg.ALLOWED_ACTIONS):
                    failed.append(j)
                else:
                    final[j] = v
            indices = failed
        for i in range(batch_size):
            if final[i] is None:
                final[i] = 'no action'
        return final

    def generate_messages(self, itv_thought, round_idx, history_batch, a_res_batch, b_res_batch, a_msg_batch, b_trust_batch, b_strategy_batch, max_retries=3):
        batch_size = len(history_batch)
        final = [None] * batch_size

        # Step 4: message_generation
        indices = list(range(batch_size))
        for attempt in range(max_retries):
            if not indices:
                break
            chats = []
            for i in range(len(indices)):
                j = indices[i]
                prompt = self.simul_cfg.PROMPTS_B['message'].format(
                    rule=self.simul_cfg.RULE,
                    round=round_idx + 1,
                    total_rounds=self.simul_cfg.NUM_ROUNDS,
                    a_res=a_res_batch[j],
                    b_res=b_res_batch[j],
                    history=history_batch[j],
                    a_msg=a_msg_batch[j],
                    trust=b_trust_batch[j],
                    strategy=b_strategy_batch[j],
                    note=self.simul_cfg.NOTE,
                )
                chats.append([{ 'role': 'user', 'content': prompt }])
                
            if itv_thought:
                itv_W_dict, itv_str_dict = dict(), dict()
                for layer in self.cfg.hook_layers:
                    itv_W = self.simul_cfg.sae_dict[layer].decoder.weight.T[self.simul_cfg.symp_label_dict[itv_thought]]
                    itv_W_batch = itv_W.repeat(len(chats), 1, 1)
                    itv_W_dict[layer] = itv_W_batch
                    
                    itv_str = torch.tensor(self.simul_cfg.layer_lambda_dict[itv_thought][layer]).to(self.simul_cfg.device_dict[layer])
                    itv_str_batch = itv_str.repeat(len(chats), 1)
                    itv_str_dict[layer] = itv_str_batch
                outputs = self.llm_generator.generate_text_w_itv(chats, itv_W_dict, itv_str_dict)
            else:
                outputs = self.llm_generator.generate_text(chats)
                
            texts = self._batch_decode(outputs)
            failed = []
            for i, t in enumerate(texts):
                j = indices[i]
                parsed = extract_and_parse_json(t)
                if 'error' in parsed or ('Agent-B message' not in parsed):
                    failed.append(j)
                else:
                    # final[j] = parsed['Agent-B message']
                    final[j] = parsed.get('Agent-B message', None)
            indices = failed
        for i in range(batch_size):
            if final[i] is None:
                final[i] = "ERROR: Failed to generate Agent-B message."
        return final


# Game mechanics
def resolve_round(a_action, b_action):

    a_action = a_action.lower()
    b_action = b_action.lower()
    
    # Both raid -> both lose all
    if a_action == 'raid' and b_action == 'raid':
        return 0, 0

    # Handle investments first
    a_invest = (a_action == 'invest')
    b_invest = (b_action == 'invest')

    a_kept = 2 - (1 if a_invest else 0)
    b_kept = 2 - (1 if b_invest else 0)

    a_gain = 0
    b_gain = 0

    # Pot resolution
    if a_invest and b_invest:
        # Pot has 2, doubles to 4, split -> 2 each
        a_gain += 2
        b_gain += 2
    elif a_invest != b_invest:
        # Single investor loses the contributed coconut
        pass

    # raid resolution (if only one raids)
    if a_action == 'raid' and b_action != 'raid':
        # A steals 1 from B's kept pile if available
        b_kept -= 1
        a_gain += 1
    if b_action == 'raid' and a_action != 'raid':
        a_kept -= 1
        b_gain += 1

    a_sc_after_action = a_kept + a_gain
    b_sc_after_action = b_kept + b_gain

    return a_sc_after_action, b_sc_after_action


def run_simulation(simul_cfg, agent_a, agent_b, ids, a_strategies, itv_thought, batch_size):
    a_history = [""] * batch_size
    b_history = [""] * batch_size
    a_total_sp = [0] * batch_size # total score points
    b_total_sp = [0] * batch_size # total score points

    for round_idx in range(simul_cfg.NUM_ROUNDS):
        
        # Phase 1: Resource distribution
        a_current_sp = [x + 2 for x in a_total_sp]
        b_current_sp = [x + 2 for x in b_total_sp]

        # Phase 2: Communication (A then B)
        if round_idx > 0:
            a_strategies = agent_a.generate_strategy(round_idx, a_history, a_current_sp, b_current_sp)
        a_msgs = agent_a.generate_messages(round_idx, a_history, a_current_sp, b_current_sp, a_strategies)
        b_trusts = agent_b.generate_analysis(itv_thought, round_idx, b_history, a_current_sp, b_current_sp, a_msgs)
        b_strategies = agent_b.generate_strategy(itv_thought, round_idx, b_history, a_current_sp, b_current_sp, a_msgs, b_trusts)
        b_msgs = agent_b.generate_messages(itv_thought, round_idx, b_history, a_current_sp, b_current_sp, a_msgs, b_trusts, b_strategies)

        # Phase 3: Actions (simultaneous)
        a_actions = agent_a.generate_actions(round_idx, a_history, a_current_sp, b_current_sp, a_strategies, a_msgs, b_msgs)
        b_actions = agent_b.generate_actions(itv_thought, round_idx, b_history, a_current_sp, b_current_sp, b_trusts, b_strategies, a_msgs, b_msgs)

        for i in range(batch_size):
            a_sp, b_sp = resolve_round(a_actions[i], b_actions[i])
            a_total_sp[i] += a_sp
            b_total_sp[i] += b_sp
            a_history[i] += (
                f"\n--- Round {round_idx + 1} ---\n"
                f"Your strategy: {a_strategies[i]}\n"
                f"Your message: {a_msgs[i]}\n"
                f"Agent-B message: {b_msgs[i]}\n"
                f"Your action: {a_actions[i]}\n"
                f"Agent-B action: {b_actions[i]}\n"
                f"You obtained: {a_sp} Coconuts this round\n"
                f"Agent-B obtained: {b_sp} Coconuts this round\n"
                f"\n--- End of Round {round_idx + 1} ---\n"
            )
            b_history[i] += (
                f"\n--- Round {round_idx + 1} ---\n"
                f"Agent-A message: {a_msgs[i]}\n"
                f"Your message: {b_msgs[i]}\n"
                f"Your trust in Agent-A: {b_trusts[i]}\n"
                f"Your strategy: {b_strategies[i]}\n"
                f"Agent-A action: {a_actions[i]}\n"
                f"Your action: {b_actions[i]}\n"
                f"Agent-A obtained: {a_sp} Coconuts this round\n"
                f"You obtained: {b_sp} Coconuts this round\n"
                f"\n--- End of Round {round_idx + 1} ---\n"
            )
            
            if simul_cfg.verbose:
                print(f"\n---------------- itv-{simul_cfg.itv_t} | Sample {ids[i]} | Round {round_idx + 1} ----------------\n")
                print(f"A strategy: {a_strategies[i]} | B strategy: {b_strategies[i]} | B trust: {b_trusts[i]} \n A action: {a_actions[i]} | B action: {b_actions[i]} \n A total: {a_total_sp[i]} | B total: {b_total_sp[i]}")
                print(f"\nAgent-A message:\n{a_msgs[i]}\n"
                      f"\nAgent-B message:\n{b_msgs[i]}\n")
                print("\n-----------------------------------------------------------------------------\n")
            

        # Save round results
        df = pd.DataFrame({
            'sample_id': ids,
            'itv_thought': [itv_thought] * batch_size,
            "itv_str": list(simul_cfg.layer_lambda_dict[itv_thought].values())[0] if itv_thought else [0.0] * batch_size,
            'round': [round_idx + 1] * batch_size,
            'a_strategy': a_strategies,
            'b_strategy': b_strategies,
            'trust_evaluation': b_trusts,            
            'agent_a_action': a_actions,
            'agent_b_action': b_actions,
            'a_total_sp': a_total_sp,
            'b_total_sp': b_total_sp,
            'agent_a_message': a_msgs,
            'agent_b_message': b_msgs,
            'agent-a': [agent_a.cfg.model_id] * batch_size,
        })
        simul_cfg.dm.save_output(df, data_type='simul_game')


def main():
    args = parameter_parser()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    from utils import Config, Data_Manager, model_selection
    
    cfg_a = Config(model_id=args.agent_a_id)
    cfg_b = Config(model_id=args.agent_b_id)

    dm = Data_Manager(cfg_b)

    model_a, tokenizer_a = model_selection(cfg_a)
    model_b, tokenizer_b = model_selection(cfg_b)
    simul_cfg = Simulation_Config(args, cfg_b, dm, model_b)

    agent_a = AgentA(cfg_a, simul_cfg, model_a, tokenizer_a)
    agent_b = AgentB(cfg_b, simul_cfg, model_b, tokenizer_b)

    sample_ids = list(range(args.num_samples))
    a_strategies = get_strategy_list(args.num_samples)

    print(f"\n--- Running Simulation: Intervention='{simul_cfg.itv_t}' ---")

    for start in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - start)
        batch_ids = sample_ids[start:start + batch_size]
        batch_strategies = a_strategies[start:start + batch_size]
        run_simulation(simul_cfg, agent_a, agent_b, batch_ids, batch_strategies, simul_cfg.itv_t, batch_size=batch_size)

if __name__ == "__main__":
    main()
