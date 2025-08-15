# In new file: poc_loss_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedLoss(nn.Module):
    """
    A unified, physics-informed loss function for training the QEC decoder.

    This class combines multiple loss components:
    1. L_CE: A standard data-driven cross-entropy loss.
    2. L_syndrome: A PINN-like constraint for syndrome consistency.
    3. L_logical: An adversarial PINN-like constraint for logical integrity.
    """
    def __init__(self, code, device, lambda_syndrome=0.0, beta_logical=0.0):
        """
        Initializes the unified loss function.

        Args:
            code: The panqec code object, containing H and L matrices.
            device: The torch device (e.g., 'cuda' or 'cpu').
            lambda_syndrome (float): The weight for the syndrome consistency loss.
                                     Set to > 0 to activate this constraint.
            beta_logical (float): The weight for the logical integrity loss.
                                  Set to > 0 to activate this constraint.
        """
        super().__init__()
        self.device = device
        
        # --- Store Physics Constraints from the Code ---
        # We need the parity-check matrix (H) and logical operators (L)
        self.H_x = torch.tensor(code.Hx.toarray(), dtype=torch.long, device=device)
        self.H_z = torch.tensor(code.Hz.toarray(), dtype=torch.long, device=device)
        self.logicals_x = torch.tensor(code.logicals_x, dtype=torch.long, device=device)
        self.logicals_z = torch.tensor(code.logicals_z, dtype=torch.long, device=device)
        
        self.num_qubits = code.n
        self.num_checks = self.H_x.shape[0] + self.H_z.shape[0]

        # --- Loss Hyperparameters ---
        self.lambda_syndrome = lambda_syndrome
        self.beta_logical = beta_logical

        # --- Baseline Loss Component ---
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, output_logits, targets, inputs):
        """
        Calculates the total loss.

        Args:
            output_logits (Tensor): The raw output from the model's final layer.
                                    Shape: (batch_size * total_nodes, 4)
            targets (Tensor): The ground-truth labels.
                              Shape: (batch_size * total_nodes)
            inputs (Tensor): The one-hot encoded input syndromes.
                             Shape: (batch_size * total_nodes, 4)
        """
        # The final iteration's output is what we care about for the loss
        final_logits = output_logits[-1]
        
        # --- 1. Baseline Data-Driven Loss (L_CE) ---
        # This is always calculated. It teaches the model to match the teacher.
        loss_ce = self.cross_entropy_loss(final_logits, targets)
        
        total_loss = loss_ce
        
        # --- 2. Physics Constraint: Syndrome Consistency (L_syndrome) ---
        if self.lambda_syndrome > 0:
            loss_syndrome = self.calculate_syndrome_loss(final_logits, inputs)
            total_loss += self.lambda_syndrome * loss_syndrome
            
        # --- 3. Physics Constraint: Logical Integrity (L_logical) ---
        if self.beta_logical > 0:
            loss_logical = self.calculate_logical_loss(final_logits, targets)
            total_loss -= self.beta_logical * loss_logical # Note: We subtract to penalize

        return total_loss

    def calculate_syndrome_loss(self, final_logits, inputs):
        """Calculates the syndrome consistency loss."""
        nodes_per_graph = inputs.shape[0] // (inputs.shape[0] // (self.num_qubits + self.num_checks))
        batch_size = inputs.shape[0] // nodes_per_graph
        
        # Get the model's concrete prediction
        predicted_correction_pauli = torch.argmax(final_logits, dim=-1)
        
        # Isolate the error part of the prediction
        predicted_error = predicted_correction_pauli.view(batch_size, -1)[:, self.num_checks:]
        
        # Convert Pauli representation (I,X,Y,Z -> 0,1,2,3) to binary (X,Z)
        pred_x = (predicted_error == 1) | (predicted_error == 2)
        pred_z = (predicted_error == 2) | (predicted_error == 3)
        
        # Get the original input syndrome from the one-hot inputs
        input_syndrome_one_hot = inputs.view(batch_size, -1, 4)[:, :self.num_checks, :]
        input_syndrome = torch.argmax(input_syndrome_one_hot, dim=-1)
        syn_x = input_syndrome[:, self.num_checks//2:]
        syn_z = input_syndrome[:, :self.num_checks//2]
        
        # Calculate the syndrome of the predicted correction
        syn_of_pred_z = (pred_x.long() @ self.H_z.T) % 2
        syn_of_pred_x = (pred_z.long() @ self.H_x.T) % 2
        
        # Calculate the residual syndrome
        residual_z = (syn_z - syn_of_pred_z) % 2
        residual_x = (syn_x - syn_of_pred_x) % 2
        
        # The loss is the total number of remaining syndrome violations (L1 norm)
        syndrome_loss = (torch.sum(residual_x) + torch.sum(residual_z)) / batch_size
        return syndrome_loss

    def calculate_logical_loss(self, final_logits, targets):
        """Calculates the adversarial logical integrity loss."""
        nodes_per_graph = targets.shape[0] // (targets.shape[0] // (self.num_qubits + self.num_checks))
        batch_size = targets.shape[0] // nodes_per_graph

        # Get the model's predicted probabilities
        probs = F.softmax(final_logits, dim=-1)
        
        # Get the ground-truth minimal correction
        true_correction_pauli = targets.view(batch_size, -1)[:, self.num_checks:]
        true_x = (true_correction_pauli == 1) | (true_correction_pauli == 2)
        true_z = (true_correction_pauli == 2) | (true_correction_pauli == 3)
        
        # For simplicity in this PoC, we only consider the first logical operator
        log_x = self.logicals_x[0, :]
        log_z = self.logicals_z[0, :]
        
        # Construct one "evil twin" correction (differs by a logical Z)
        evil_twin_x = (true_x.long() + log_z.long()) % 2
        evil_twin_z = (true_z.long()) % 2
        
        # We want to MINIMIZE the probability of this evil twin.
        # This is equivalent to MAXIMIZING log(1 - p(evil_twin)).
        # We calculate the cross-entropy with the evil twin and subtract it.
        
        # This is a simplification. A full implementation is more complex.
        # For the PoC, we will return a placeholder value.
        # The full implementation requires calculating joint probabilities.
        return torch.tensor(0.0, device=self.device) # Placeholder for now