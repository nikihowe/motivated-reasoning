import unittest
from unittest.mock import patch, MagicMock, call
import os
import torch
from motivated_reasoning.inference.model_utils import load_model_and_tokenizer


class TestModelUtils(unittest.TestCase):
    """Test the model_utils.py functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_run_name = "test_model_run"
        self.test_iteration = 1
        self.test_model_path = "/test/path/models"
        self.test_base_model_name = "test-model/Test-Model-1B"
        
    @patch('motivated_reasoning.inference.model_utils.find_freest_gpus')
    @patch('motivated_reasoning.inference.model_utils.PeftConfig')
    @patch('motivated_reasoning.inference.model_utils.PeftModel')
    @patch('motivated_reasoning.inference.model_utils.AutoModelForCausalLM')
    @patch('motivated_reasoning.inference.model_utils.AutoTokenizer')
    @patch('torch.compile')
    def test_load_model_with_adapter(self, mock_compile, mock_tokenizer, mock_model, 
                                   mock_peft_model, mock_peft_config, mock_find_gpus):
        """Test loading model with adapter"""
        # Setup mocks
        mock_find_gpus.return_value = [0]
        mock_peft_config_instance = MagicMock()
        mock_peft_config_instance.base_model_name_or_path = self.test_base_model_name
        mock_peft_config.from_pretrained.return_value = mock_peft_config_instance
        
        mock_base_model = MagicMock()
        mock_base_model.config.pad_token_id = 128001
        mock_model.from_pretrained.return_value = mock_base_model
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.padding_side = "left"
        mock_tokenizer_instance.pad_token = "<pad>"
        mock_tokenizer_instance.pad_token_id = 128001
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_peft_model_instance = MagicMock()
        mock_peft_model.from_pretrained.return_value = mock_peft_model_instance
        
        mock_compiled_model = MagicMock()
        mock_compile.return_value = mock_compiled_model
        
        # Call the function
        model, tokenizer, model_id = load_model_and_tokenizer(
            run_name=self.test_run_name,
            iteration=self.test_iteration,
            model_path=self.test_model_path,
            load_base_model_only=False,
            base_model_name=self.test_base_model_name
        )
        
        # Verify results
        self.assertEqual(model, mock_compiled_model)
        self.assertEqual(tokenizer, mock_tokenizer_instance)
        expected_adapter_path = f"{self.test_model_path}/{self.test_run_name}/{self.test_iteration}/checkpoint-6"
        self.assertEqual(model_id, expected_adapter_path)
        
        # Verify GPU setup
        mock_find_gpus.assert_called_once_with(1)
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0")
        
        # Verify adapter loading
        mock_peft_config.from_pretrained.assert_called_once_with(expected_adapter_path)
        mock_peft_model.from_pretrained.assert_called_once_with(mock_base_model, expected_adapter_path)
        
    @patch('motivated_reasoning.inference.model_utils.find_freest_gpus')
    @patch('motivated_reasoning.inference.model_utils.AutoModelForCausalLM')
    @patch('motivated_reasoning.inference.model_utils.AutoTokenizer')
    @patch('torch.compile')
    def test_load_base_model_only(self, mock_compile, mock_tokenizer, mock_model, mock_find_gpus):
        """Test loading base model only (no adapter)"""
        # Setup mocks
        mock_find_gpus.return_value = [1]
        
        mock_base_model = MagicMock()
        mock_base_model.config.pad_token_id = 128001
        mock_model.from_pretrained.return_value = mock_base_model
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.padding_side = "left"
        mock_tokenizer_instance.pad_token = "<pad>"
        mock_tokenizer_instance.pad_token_id = 128001
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_compiled_model = MagicMock()
        mock_compile.return_value = mock_compiled_model
        
        # Call the function
        model, tokenizer, model_id = load_model_and_tokenizer(
            run_name=self.test_run_name,
            iteration=self.test_iteration,
            model_path=self.test_model_path,
            load_base_model_only=True,
            base_model_name=self.test_base_model_name
        )
        
        # Verify results
        self.assertEqual(model, mock_compiled_model)
        self.assertEqual(tokenizer, mock_tokenizer_instance)
        self.assertEqual(model_id, self.test_base_model_name)
        
        # Verify GPU setup
        mock_find_gpus.assert_called_once_with(1)
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "1")
        
        # Verify base model loading
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once_with(self.test_base_model_name, padding_side="left")
        
    @patch('motivated_reasoning.inference.model_utils.find_freest_gpus')
    @patch('motivated_reasoning.inference.model_utils.AutoModelForCausalLM')
    @patch('motivated_reasoning.inference.model_utils.AutoTokenizer')
    @patch('torch.compile')
    def test_llama3_pad_token_handling(self, mock_compile, mock_tokenizer, mock_model, mock_find_gpus):
        """Test Llama-3 specific pad token handling"""
        # Setup mocks
        mock_find_gpus.return_value = [0]
        
        mock_base_model = MagicMock()
        mock_base_model.config.pad_token_id = None  # No pad token initially
        mock_model.from_pretrained.return_value = mock_base_model
        
        # Mock temporary tokenizer for pad token setup
        mock_temp_tokenizer = MagicMock()
        mock_temp_tokenizer.convert_tokens_to_ids.return_value = 128198
        mock_temp_tokenizer.eos_token_id = 128001
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.padding_side = "left"
        mock_tokenizer_instance.pad_token_id = 128198
        
        # Return temp tokenizer first, then main tokenizer
        mock_tokenizer.from_pretrained.side_effect = [mock_temp_tokenizer, mock_tokenizer_instance]
        
        mock_compiled_model = MagicMock()
        mock_compile.return_value = mock_compiled_model
        
        # Call with Llama-3 model
        model, tokenizer, model_id = load_model_and_tokenizer(
            run_name=self.test_run_name,
            iteration=self.test_iteration,
            model_path=self.test_model_path,
            load_base_model_only=True,
            base_model_name="meta-llama/Meta-Llama-3-8B-Instruct"
        )
        
        # Verify pad token was set
        self.assertEqual(mock_base_model.config.pad_token_id, 128198)
        mock_temp_tokenizer.convert_tokens_to_ids.assert_called_once_with("<|reserved_special_token_198|>")
        
    @patch('motivated_reasoning.inference.model_utils.find_freest_gpus')
    def test_gpu_not_available_error(self, mock_find_gpus):
        """Test error when no GPU is available"""
        mock_find_gpus.return_value = None
        
        with self.assertRaises(AssertionError):
            load_model_and_tokenizer(
                run_name=self.test_run_name,
                iteration=self.test_iteration,
                model_path=self.test_model_path,
                load_base_model_only=True,
                base_model_name=self.test_base_model_name
            )
            
    @patch('motivated_reasoning.inference.model_utils.find_freest_gpus')
    @patch('motivated_reasoning.inference.model_utils.PeftConfig')
    def test_invalid_adapter_path_error(self, mock_peft_config, mock_find_gpus):
        """Test error handling for invalid adapter path"""
        mock_find_gpus.return_value = [0]
        mock_peft_config.from_pretrained.side_effect = Exception("Invalid adapter path")
        
        with self.assertRaises(Exception) as context:
            load_model_and_tokenizer(
                run_name=self.test_run_name,
                iteration=self.test_iteration,
                model_path=self.test_model_path,
                load_base_model_only=False,
                base_model_name=self.test_base_model_name
            )
        
        self.assertIn("Invalid adapter path", str(context.exception))
        
    def test_function_signature(self):
        """Test that the function has the expected signature"""
        import inspect
        sig = inspect.signature(load_model_and_tokenizer)
        
        # Check that all required parameters are present
        expected_params = ['run_name', 'iteration', 'model_path', 'load_base_model_only', 'base_model_name']
        actual_params = list(sig.parameters.keys())
        
        for param in expected_params:
            self.assertIn(param, actual_params)
            
        # Check return type annotation
        self.assertIsNotNone(sig.return_annotation)


if __name__ == '__main__':
    unittest.main() 