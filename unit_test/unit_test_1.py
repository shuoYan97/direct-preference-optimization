import torch
import random
import unittest
from trainers import preference_loss

class TestModelOutputs(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # Load inputs and outputs from .pt files
        self.policy_chosen_logps = torch.load("tensors/policy_chosen_logps.pt").to(self.device)  
        self.policy_rejected_logps = torch.load("tensors/policy_rejected_logps.pt").to(self.device)  
        self.reference_chosen_logps = torch.load("tensors/reference_chosen_logps.pt").to(self.device)  
        self.reference_rejected_logps = torch.load("tensors/reference_rejected_logps.pt").to(self.device)  
        # self.reference_free = False
        self.beta = 0.1
        # self.label_smoothing = 0
        print("policy_chosen_logps:", self.policy_chosen_logps)
        print("policy_rejected_logps:", self.policy_rejected_logps)
        print("reference_chosen_logps:", self.reference_chosen_logps)
        print("reference_rejected_logps:", self.reference_rejected_logps)


        
        self.losses = torch.load("tensors/losses.pt").to(self.device)  
        # self.chosen_rewards = torch.load("tensors/chosen_rewards.pt").to(self.device)  
        # self.rejected_rewards = torch.load("tensors/rejected_rewards.pt").to(self.device)  

        # Ensure inputs and outputs have the same length
        assert len(self.policy_chosen_logps) == len(self.losses), "Inputs and outputs must have the same length"
        print("len:", len(self.losses))

    def test_random_pairs(self):
        # Select three random indices
        # indices = random.sample(range(len(self.policy_chosen_logps)), 3)
        indices = [0, 1]

        for idx in indices:
            policy_chosen_logps = self.policy_chosen_logps[idx]
            policy_rejected_logps = self.policy_rejected_logps[idx]
            reference_chosen_logps = self.reference_chosen_logps[idx]
            reference_rejected_logps = self.reference_rejected_logps[idx]

            losses_expected = self.losses[idx]
            # chosen_rewards_expected = self.chosen_rewards[idx]
            # rejected_rewards_expected = self.rejected_rewards[idx]

            losses, chosen_rewards, rejected_rewards = preference_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=self.beta)
            
            self.assertTrue(torch.allclose(losses, losses_expected, atol=1e-5), f"Losses Expected {losses_expected}, got {losses}")


if __name__ == "__main__":
    unittest.main()
