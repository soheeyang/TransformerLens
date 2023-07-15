import pytest
import torch
from transformer_lens import utils

from transformer_lens import HookedTransformer


class TestPosEmbedWithLeftPadding:

    prompts = [
        'Hello world!',
        'How are you today?',
        'I\'m fine, thank you.',
        'I am happy.',
    ]
    
    def get_logits(self, model, prompts, run_with_cache):
        if run_with_cache:
            return model.run_with_cache(prompts)[0]
        else:
            return model(prompts)

    # fixtures
    @pytest.fixture(scope="class", params=["gpt2-small", "facebook/opt-125m"])
    def model_name(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def model(self, model_name):
        model = HookedTransformer.from_pretrained(model_name)
        return model
    
    # tests
    @pytest.mark.parametrize("padding_side", ["left", "right"])
    def test_pos_embed(self, model, padding_side):
        # setup
        model.tokenizer.padding_side = padding_side

        prompts = self.prompts
        tokens = model.to_tokens(prompts)
        str_tokens = model.to_str_tokens(prompts)
        
        left_attention_mask = utils.get_attention_mask(
            model.tokenizer, tokens
        )  # [batch pos]
        
        output_pos_embed = model.pos_embed(tokens, 0, left_attention_mask=left_attention_mask)  # [batch pos d_model]
        
        # check if the output pos_embeds have the correct shape
        assert output_pos_embed.shape == (tokens.shape[0], tokens.shape[1], model.pos_embed.W_pos.shape[1])
        
        # check if the target pos_embeds are the same as the output pos_embeds
        target_position_ids = torch.tensor(sum([list(range(len(t))) for t in str_tokens], []), device=tokens.device)
        target_output_pos_embed = model.pos_embed.W_pos[target_position_ids, :]
        
        attended_output_pos_embed = output_pos_embed[left_attention_mask.bool()]

        assert torch.allclose(attended_output_pos_embed, target_output_pos_embed)
        
        # padded positions should have zero pos_embed
        assert output_pos_embed[~left_attention_mask.bool()].sum() == 0
        
    @pytest.mark.parametrize("run_with_cache", [True, False])
    def test_left_padding(self, model, run_with_cache):
        prompts = self.prompts
        
        num_str_tokens_list = [len(t) for t in model.to_str_tokens(prompts)]
        
        # left padding output
        model.tokenizer.padding_side = "left"
        left_logits = self.get_logits(model, prompts, run_with_cache=run_with_cache)
        left_last_logits = left_logits[:, -1, :]
        left_first_token_positions = left_logits.shape[1] - torch.tensor(num_str_tokens_list, device=left_logits.device)
        left_first_logits = left_logits[torch.arange(len(prompts)), left_first_token_positions, :].squeeze(1)
        
        # right padding output
        model.tokenizer.padding_side = "right"
        right_logits = self.get_logits(model, prompts, run_with_cache=run_with_cache)
        right_last_token_positions = torch.tensor(num_str_tokens_list, device=right_logits.device) - 1
        right_last_logits = right_logits[torch.arange(len(prompts)), right_last_token_positions, :].squeeze(1)        
        right_first_logits = right_logits[:, 0, :]
        
        # check if the left and right padding outputs are the same for the first and last tokens
        assert torch.allclose(left_last_logits, right_last_logits)
        assert torch.allclose(left_first_logits, right_first_logits)
        
        # check if the left and right padding outputs are the same for all tokens
        # and if the batched padded outputs are the same as the single prompt outputs
        right_token_start = 0
        left_token_end = left_logits.shape[1]
        for i, (prompt, left_token_start, right_token_end) in enumerate(zip(
            prompts, left_first_token_positions.tolist(), (right_last_token_positions + 1).tolist()
        )):
            single_logits = self.get_logits(model, prompt, run_with_cache=run_with_cache)[0]
            
            assert right_token_end - right_token_start == left_token_end - left_token_start == single_logits.shape[0]

            assert torch.allclose(
                left_logits[i, left_token_start:left_token_end, :],
                right_logits[i, right_token_start:right_token_end, :],
            )
            
            assert torch.allclose(
                left_logits[i, left_token_start:left_token_end, :],
                single_logits,
                atol=1e-4,
            )
            
            assert torch.allclose(
                right_logits[i, right_token_start:right_token_end, :],
                single_logits,
                atol=1e-4,
            )