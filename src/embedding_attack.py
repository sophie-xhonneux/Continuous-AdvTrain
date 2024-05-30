import torch
from torch.optim.optimizer import Optimizer
import logging

INIT_TYPES = ["instruction", "suffix"]


class NoAttack:
    def __init__(
        self,
        embed_weights,
    ) -> None:
        """
        No Attack
        """
        self.embed_weights = embed_weights
        self.vocab_size = self.embed_weights.shape[0]
        self.embedding_size = self.embed_weights.shape[1]
        return

    def attack(self, model, input_ids, target_ids, attention_mask):
        all_losses = []
        affirmative_responses = []

        adv_perturbation, adv_perturbation_mask = self.init_perturbation(
            input_ids, target_ids, attention_mask
        )
        input_embeds = self.get_embeddings(input_ids)

        return (
            input_embeds.detach(),
            adv_perturbation.detach(),
            adv_perturbation_mask.detach(),
            all_losses,
            affirmative_responses,
        )

    def get_adv_embeddings(self, input_embeds, adv_perturbation, adv_perturbation_mask):
        return input_embeds

    def init_perturbation(self, input_ids, target_ids, attention_mask):
        target_mask = target_ids > 0
        input_mask = (~target_mask * attention_mask).to(bool)
        batch_size, num_input_tokens = input_ids.shape
        dtype = self.embed_weights.dtype

        adv_perturbation = torch.zeros(
            (batch_size, num_input_tokens, self.embedding_size), device=input_ids.device, dtype=dtype
        )

        adv_perturbation_mask = torch.zeros(
            (batch_size, num_input_tokens), device=input_ids.device, dtype=input_ids.dtype
        )
        adv_perturbation_mask[input_mask] = 1

        return adv_perturbation, adv_perturbation_mask.unsqueeze(-1)

    def get_one_hot(self, ids):
        device = self.embed_weights.device
        batch_size, num_tokens = ids.shape

        # Adjusting IDs less than 0 to 0
        ids = torch.where(ids < 0, torch.tensor(0, device=device, dtype=ids.dtype), ids)

        one_hot = torch.zeros(
            batch_size, num_tokens, self.vocab_size, device=device, dtype=self.embed_weights.dtype
        )
        one_hot.scatter_(2, ids.unsqueeze(2), 1)
        return one_hot

    def get_embeddings(self, ids):
        one_hot = self.get_one_hot(ids)
        embeddings = (one_hot @ self.embed_weights).data
        return embeddings


class EmbeddingSpaceAttack:
    def __init__(
        self,
        embed_weights,
        response_key,
        tokenizer,
        iters=8,
        opt_config=None,
        eps=0.01,
        init_type="instruction",
        suffix_tokens=10,
        relative_lr=False,
        debug=0,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the EmbeddingAttack class.

        Args:
            embed_weights (torch.Tensor): The token embedding matrix of the respective model.
            response_key (str): The response key of the datacollator used to separate instructions and targets.
            tokenizer (Tokenizer): The tokenizer object.
            iters (int, optional): The number of iterations.
            opt_config (dict, optional): The optimizer configuration.
            eps (float, optional): The epsilon value.
            init_type (str, optional): The initialization type
            suffix_tokens (int, optional): The number of suffix tokens
            debug (int, optional): The debug level with increasing verbosity. (0 None, 1 print final loss, 2 include loss at every iteration, 3 include norms and shapes)
        """

        self.embed_weights = embed_weights
        # self.response_key = response_key  # TODO Needed? Remove if not (also line bellow and argument)
        # self.response_key_ids = tokenizer.encode(response_key, return_tensors="pt")
        self.tokenizer = tokenizer
        self.vocab_size = self.embed_weights.shape[0]
        self.embedding_size = self.embed_weights.shape[1]
        self.embedding_norm = torch.norm(embed_weights, p=2, dim=-1).mean()
        self.iters = iters
        self.opt_config = opt_config
        if relative_lr:
            self.opt_config["lr"] = self.opt_config["lr"] * self.eps
        self.eps = eps * self.embedding_norm

        logging.info(
            f"L2 norm of embedding weights equals {self.embedding_norm} eps multiplier is: {eps} using eps: {self.eps}"
        )

        if init_type not in INIT_TYPES:
            ValueError(f"init_type must be in {INIT_TYPES} and not {self.init_type}")
        self.init_type = init_type

        self.suffix_tokens = suffix_tokens
        self.debug = debug
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def attack(self, model, input_ids, target_ids, attention_mask):
        # save losses and responses
        best_loss = torch.inf
        all_losses = []
        affirmative_responses = []

        # init embeddings of input instruction and target and initialize adversarial perturbation
        adv_perturbation, adv_perturbation_mask = self.init_perturbation(
            input_ids, target_ids, attention_mask
        )
        input_embeds = self.get_embeddings(input_ids)
        target_one_hot = self.get_one_hot(target_ids)

        # get attention mask. If we add an adversarial suffic we need to adapt it
        attention_mask = self.get_attention_mask(input_ids, attention_mask)

        # loss targets
        loss_mask = self.get_loss_mask(target_ids)

        # init opt
        opt = self.init_opt([adv_perturbation])

        if self.debug > 2:
            self.debug_shapes(
                input_embeds,
                target_one_hot,
                adv_perturbation,
                adv_perturbation_mask,
                attention_mask,
                loss_mask,
            )

        for i in range(self.iters):
            opt.zero_grad()
            adv_embeds = self.get_adv_embeddings(input_embeds, adv_perturbation, adv_perturbation_mask)
            logits, loss = self.calc_loss(i, model, adv_embeds, target_one_hot, attention_mask, loss_mask)
            loss.backward()
            all_losses.append(loss.item())
            opt.step()

            # perturbations that are applied directly to the instruction are constraint to a small lp ball
            if self.init_type == "instruction":
                adv_perturbation = self.project_l2(adv_perturbation)
            # perturbations that are added as a suffix should be projected to the simplex
            elif self.init_type == "suffix":
                adv_perturbation = self.project_simplex(adv_perturbation)

            num_affirmative_responses = self.get_num_affirmative_responses(target_ids, logits)
            affirmative_responses.append(num_affirmative_responses)

            # save best perturbation
            if loss < best_loss:
                best_loss = loss
                self.best_adv_perturbation = adv_perturbation

            # debug
            if self.debug > 2:
                self.debug_norm(adv_perturbation)

        if self.debug > 0:
            self.debug_loss(best_loss)
        if self.debug > 2:
            self.debug_output(target_ids, logits, attention_mask)

        return (
            input_embeds.detach(),
            adv_perturbation.detach(),
            adv_perturbation_mask.detach(),
            all_losses,
            affirmative_responses,
        )

    def calc_loss(self, i, model, input_embeds, target_one_hot, attention_mask, loss_mask, log_debug=True):
        output = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        logits = output.logits  # ignore predicted token (not in targets)

        logits_loss_mask = logits[:, :-1][loss_mask]
        target_one_hot_loss_mask = target_one_hot[:, 1:][loss_mask]  # ignore first token (not in predicted)
        loss = self.loss_fct(logits_loss_mask, target_one_hot_loss_mask)

        if i == 0:
            self.logits_benign = logits
            self.loss_benign = loss

        if self.debug > 1 and log_debug:
            self.debug_iter_loss(i, loss)

        return logits, loss

    def project_l2(self, adv_perturbation):
        norm = torch.norm(adv_perturbation, p=2, dim=-1, keepdim=True)
        mask = (norm > self.eps).squeeze()
        if torch.any(mask):
            with torch.no_grad():
                if len(mask.shape) == 1:  # batch size 1
                    mask = mask.unsqueeze(0)
                adv_perturbation[mask, :] = adv_perturbation[mask, :] / norm[mask] * self.eps

        return adv_perturbation

    def project_simplex(self, adv_perturbation):
        # TODO --> suffix perturbations should be optimized in the one-hot space and not at the embedding space
        raise NotImplementedError("Simplex projection not implemented yet")

    def get_one_hot(self, ids):
        device = self.embed_weights.device
        batch_size, num_tokens = ids.shape

        # Adjusting IDs less than 0 to 0
        ids = torch.where(ids < 0, torch.tensor(0, device=device, dtype=ids.dtype), ids)

        one_hot = torch.zeros(
            batch_size, num_tokens, self.vocab_size, device=device, dtype=self.embed_weights.dtype
        )
        one_hot.scatter_(2, ids.unsqueeze(2), 1)
        return one_hot

    def get_embeddings(self, ids):
        one_hot = self.get_one_hot(ids)
        embeddings = (one_hot @ self.embed_weights).data
        return embeddings

    def get_adv_embeddings(self, input_embeds, adv_perturbation, adv_perturbation_mask):
        masked_perturbation = adv_perturbation * adv_perturbation_mask
        adv_embeds = input_embeds + masked_perturbation
        return adv_embeds

    def get_loss_slice_start_and_end(self, input_embeds):
        input_len = input_embeds.shape[1]

        if self.init_type == "instruction":
            start = input_len - 1
            end = -1
        elif self.init_type == "suffix":
            start = input_len + self.suffix_tokens
            end = -1
        return start, end

    def get_attention_mask(self, input_ids, attention_mask):
        if self.init_type == "instruction":
            return attention_mask
        elif self.init_type == "suffix":
            len_input = input_ids.shape[1]
            input_mask = attention_mask[:, :len_input]
            adversarial_mask = torch.ones(
                (attention_mask.shape[0], self.suffix_tokens),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            target_mask = attention_mask[:, len_input:]
            attention_mask = torch.hstack([input_mask, adversarial_mask, target_mask])
            return attention_mask

    def get_loss_mask(self, target_ids):
        target_mask = target_ids > 0
        if self.init_type == "instruction":
            return target_mask[:, 1:]  # ignore first token (not in predicted)
        elif self.init_type == "suffix":
            padding_for_suffix = torch.zeros(
                (target_mask.shape[0], self.suffix_tokens), dtype=target_ids.dtype, device=target_ids.device
            )
            loss_mask = torch.hstack([padding_for_suffix, target_mask])
        return loss_mask

    def init_perturbation(self, input_ids, target_ids, attention_mask):
        target_mask = target_ids > 0
        input_mask = (~target_mask * attention_mask).to(bool)
        batch_size, num_input_tokens = input_ids.shape
        dtype = self.embed_weights.dtype

        if self.init_type == "instruction":
            adv_perturbation = torch.zeros(
                (batch_size, num_input_tokens, self.embedding_size), device=input_ids.device, dtype=dtype
            )

            adv_perturbation_mask = torch.zeros(
                (batch_size, num_input_tokens), device=input_ids.device, dtype=input_ids.dtype
            )
            adv_perturbation_mask[input_mask] = 1
        elif self.init_type == "suffix":
            adv_perturbation = torch.randn(
                (batch_size, num_input_tokens + self.suffix_tokens, self.embedding_size),
                device=input_ids.device,
                dtype=dtype,
            )
            adv_perturbation = self.project_simplex(adv_perturbation)

            adv_perturbation_mask = torch.zeros(
                (batch_size, num_input_tokens + self.suffix_tokens),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            # get indexes of suffix tokens and set them to one
            num_false = torch.sum(input_mask, dim=1)  # TODO FIX as input mask changed
            row_indices = torch.arange(adv_perturbation_mask.shape[0])
            col_indices = num_false[:, None] + torch.arange(self.suffix_tokens)
            col_indices = torch.clip(col_indices, 0, adv_perturbation_mask.shape[1] - 1)
            adv_perturbation_mask[row_indices[:, None], col_indices] = True

        adv_perturbation.requires_grad = True
        adv_perturbation_mask = adv_perturbation_mask.unsqueeze(2)

        # save best adv_perturbation
        self.best_adv_perturbation = adv_perturbation

        return adv_perturbation, adv_perturbation_mask

    def init_opt(self, parameters):
        if self.opt_config is None:
            self.opt_config = {"type": "sign", "lr": 0.01}
            logging.info(f"No opt_config specified using default opt_config: {self.opt_config}")

        optimizer_type = self.opt_config["type"]
        if optimizer_type == "adam":
            opt = torch.optim.Adam(parameters, lr=self.opt_config["lr"])
        elif optimizer_type == "sign":
            opt = SignSGD(parameters, lr=self.opt_config["lr"])
        elif optimizer_type == "rms":
            opt = torch.optim.RMSprop(parameters, lr=self.opt_config["lr"])

        return opt

    def get_num_affirmative_responses(self, target_ids, logits):  # TODO Fix for suffix attack
        target_ids_clone = target_ids.clone()
        target_ids_clone = target_ids_clone[:, 1:]  # ignore first token (not in predicted)
        output_ids = torch.argmax(logits, dim=-1)
        output_ids = output_ids[:, :-1]  # ignore predicted token (not in target_ids)

        input_mask = target_ids_clone < 0
        output_ids[input_mask] = 0
        target_ids_clone[input_mask] = 0
        affirmative_responses = output_ids == target_ids_clone
        affirmative_responses_sum = affirmative_responses.all(dim=-1).sum().item()

        return affirmative_responses_sum

    def debug_norm(self, adv_perturbation):
        if self.init_type == "instruction":
            norm = torch.norm(adv_perturbation, p=2, dim=-1).max()
            logging.info(f"Debugging ESA | L2 Norm max adversarial perturbation: {norm}")
        # TODO print norm for simplex projection?

    def debug_loss(self, loss_adversarial):
        logging.info(
            f"Debugging ESA | Benign loss: {self.loss_benign.item()} | Best Adversarial loss {loss_adversarial.item()}"
        )

    def debug_iter_loss(self, i, loss_adversarial):
        logging.info(f"Debugging ESA | i: {i} | Adversarial loss: {loss_adversarial.item()}")

    def debug_shapes(
        self, input_embeds, target_one_hot, adv_perturbation, adv_perturbation_mask, attention_mask, loss_mask
    ):
        logging.info(
            "====== Debugging ESA Adversarial Attack Shapes ======\n"
            f"input_embeds: {input_embeds.shape}\n"
            f"adv_perturbation: {adv_perturbation.shape}\n"
            f"adv_perturbation_mask: {adv_perturbation_mask.shape}\n"
            f"target_one_hot: {target_one_hot.shape}\n"
            f"attention_mask: {attention_mask.shape}\n"
            f"loss_mask: {loss_mask.shape}"
        )

    def debug_output(self, target_ids, logits, attention_mask):
        with torch.no_grad():
            attention_mask = attention_mask.to(dtype=torch.bool)

            # only first sample; ignore first token (not in predicted tokens)
            target_ids = target_ids[0][1:]
            target_mask = target_ids > 0
            only_targets = target_ids[target_mask]
            original_text = self.tokenizer.decode(only_targets, skip_special_tokens=True)

            output_ids_benign = torch.argmax(self.logits_benign, dim=-1)
            # only first sample; only the target tokens (model reply); remove predicted token at the end
            output_ids_benign = output_ids_benign[0][:-1][target_mask]
            generated_text_benign = self.tokenizer.decode(output_ids_benign, skip_special_tokens=True)

            output_ids_adv = torch.argmax(logits, dim=-1)
            output_ids_adv = output_ids_adv[0][:-1][target_mask]
            generated_text_adv = self.tokenizer.decode(output_ids_adv, skip_special_tokens=True)
            logging.info(
                "===== Debugging ESA Original text ====\n"
                f"{original_text}\n"
                "===== Debugging ESA Generated text benign ====\n"
                f"{generated_text_benign}\n"
                "===== Debugging ESA Generated text adversarial ====\n"
                f"{generated_text_adv}"
            )


class SignSGD(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(SignSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    sign = torch.sign(grad)

                    p.add_(other=sign, alpha=-group["lr"])

        return loss
