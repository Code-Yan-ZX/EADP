@torch.no_grad()
def forward(self, images, texts=None, return_attentions=False):
    if type(images) is list:
        image_features = []
        for image in images:
            image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
            image_feature = self.feature_select(image_forward_out).to(image.dtype)
            image_features.append(image_feature)
    else:
        if return_attentions and texts is None:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            return (image_features, image_forward_outs.attentions)
        image_stream = torch.cuda.Stream()
        text_stream = torch.cuda.Stream()
        caller_stream = torch.cuda.current_stream(device=self.device)
        image_stream.wait_stream(caller_stream)
        text_stream.wait_stream(caller_stream)
        with torch.cuda.stream(image_stream):
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_outputs = self.feature_select(image_forward_outs)
            image_features = image_outputs.to(images.dtype)
        if texts is not None:
            with torch.cuda.stream(text_stream):
                text_inputs = self.text_tokenizer(text=texts, return_tensors='pt')
                text_segment = (text_inputs.input_ids.shape[1] - 1) // self.max_position_embeddings + 1
                text_padding = self.max_position_embeddings * text_segment - text_inputs.input_ids.shape[1]
                text_inputs = {k: torch.cat([v, v.new_zeros((v.shape[0], text_padding))], dim=1).reshape(-1, self.max_position_embeddings).to(device=self.device) for k, v in text_inputs.items()}
                outputs = self.text_tower(**text_inputs)
                text_embeds = outputs.text_embeds
                text_embeds_seq = outputs.last_hidden_state
                if hasattr(self.text_tower, 'text_projection'):
                    text_embeds_seq = self.text_tower.text_projection(text_embeds_seq)
        torch.cuda.synchronize()
        if texts is not None:
            image_embeds = self.vision_tower.vision_model.post_layernorm(image_outputs)
            proj_dtype = self.vision_tower.visual_projection.weight.dtype
            image_embeds = self.vision_tower.visual_projection(image_embeds.to(proj_dtype))
            image_features = (image_features, image_embeds, text_embeds, text_embeds_seq)
    return image_features
