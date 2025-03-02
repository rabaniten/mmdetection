# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)


ALL_LABELS = ('chervil cream sauce', 'small sauce-glass', 'cream tart', 'white plate', 'chicken breast', 'tortellini', 'shallow bowl', 'cherry tomato', 'caesar salad', 'parmesan dressing', 'egg cooked', 'toast croutons', 'parmesan shavings', 'bacon', 'lettuce', 'dressing', 'colorful vegetable pan with soft egg noodles (spaetzle)', 'white plate without rim', 'capuns', 'soup of the day ratatouille cream', 'root vegetables', 'soup-bowl', 'beer sauce', 'finger-shaped potato dumplings (schupfnudeln)', 'sauerkraut', 'smoked pork neck', 'roasted cashew nuts', 'hioumi', 'raspberry quark', 'natural yogurt', 'strawberry yogurt', 'apricot yogurt', 'raspberry yogurt', 'beans green', 'cream sauce', 'vegetable bolognese', 'potatoes', 'thyme', 'horn-shaped pasta (hoernli)', 'caramel flan', 'big sauce-glass', 'fruit salad', 'soup of the day artichoke', 'veal sausage', 'onion sauce', 'glass fruitsalad-bowl', 'spanish tortilla', 'basil sauce', 'vanilla cream puffs', 'small quadratic plate-bowl', 'basil pesto', 'quadratic dessert-plate', 'radish', 'sausage and cheese salad', 'lollo bianco', 'house bread', 'soup of the day potato', 'vegetarian bami goreng', 'lollo rosso', 'soup of the day broccoli cream', 'spaghetti', 'vegetable strips', 'saffron herb sauce', 'vegetable strips saffron-herb sauce', 'pureed green balls', 'pureed food in a special shape', 'eggplant moussaka', 'pureed meat slices', 'pureed food in oval shape', 'pureed broccoli', 'pureed mashed potatoes', 'spinach tart', 'turmeric', 'pita bread', 'trout fillet', 'spinach', 'rice', 'sauce', 'sprout vegetables', 'scallion', 'herbal rice', 'soup of the day curry cream', 'lentil ragout', 'paneer', 'quinoa patties', 'vegetables for quinoa patties', 'toast', 'turkey ham', 'pineapple', 'gruyere cheese', 'onion red', 'cranberry', 'barley risotto', 'salad leaves', 'cheese tart', 'zucchetti', 'lemon panna cotta', 'parsley fritters', 'lemon', 'capers', 'smoked trout', 'trout tartare', 'soup of the day yellow pea', 'horseradish foam', 'butter', 'vegetable lasagna', 'applesauce', 'broccoli', 'dill mashed potatoes', 'salmon cubes marinated', 'white wine sauce', 'brown sauce', 'pureed food in pyramid shape', 'penne rigate', 'pureed polenta', 'pureed balls', 'tagliatelle', 'pureed cauliflower', 'raspberry', 'raspberry mousse in pyramid shape', 'bolognese', 'white sauce', 'bell pepper sauce', 'chocolate mousse', 'round raspberry mousse', 'romanesco', 'slices', 'poultry stew', 'pureed salmon', 'pureed chicken thigh', 'pureed fries', 'pureed sausage', 'penne', 'plate with red rim', 'soft egg noodles (spaetzle)', 'veggie swiss macaroni and cheese', 'poulet', 'pasta', 'boiled meat salad seed oil', 'vegetable strudel', 'vegetable patch', 'herb quark dip', 'vegetables for boiled meat salad', 'plum tart', 'boiled meat', 'gnocchi seitan pan', 'lamb stew', 'bulgur', 'carrots', 'currant sheet cake', 'bulgur sauce', 'gnocchi', 'sliced seitan', 'oyster mushrooms', 'vegetable salad with white beans', 'vegetables for green spelt risotto', 'green spelt risotto', 'rye bread', 'apple tart', 'bouillon', 'cold chicken breast', 'cocktail sauce', 'soup of the day carrot cream', 'curry dip', 'soup of the day lentil ginger', 'poulet cordon bleu', 'jus', 'pilau rice', 'sugar peas', 'roasted cauliflower', 'sauce for sliced seitan', 'boiled potatoes', 'semolina porridge', 'cherry compote', 'cinnamon sugar', 'small plastic cup', 'oversoaked sliced chicken', 'overly soft thick brie cheese', 'overly soft cottage cheese', 'meat cheese', 'mustard sauce', 'lyonnaise potatoes', 'oversoaked sliced veal', 'overly soft cream cheese', 'overly soft thin brie cheese', 'currants', 'soup of the day bell peppers', 'sliced quorn sauce zurich style', 'sliced quorn zurich style', 'colorful vegetables from zuchetti peas carrots and beans', 'hash brown (roesti)', 'bread dumplings', 'sauce poultry ragout', 'sliced quorn', 'vegetables', 'banana organic', 'croissant', 'lye croissant', 'coffee', 'lid on the ground', 'uncovered jug', 'jug covered with lid', 'orange juice', 'multigrain roll', 'scrambled eggs', 'milk', 'mueesli', 'large glass fruitsalad-bowl', 'milk roll', 'mozzarella', 'baked vegetables for mozzarella', 'wedges', 'chipolata sausage', 'vegetables for chipolata sausage', 'pepper', 'rucola', 'walnut', 'mozzarella salad', 'oven vegetables', 'zuchetti', 'eggplant', 'piccata mass', 'bramata slice', 'vegetables for piccata', 'cream', 'chocolate cake', 'fried rice', 'beef meatballs', 'spicy vegetable ragout', 'olives', 'cucumber', 'tomato', 'turkey breast', 'carrot', 'chili with vegetables', 'lenses brown', 'lenses', 'pear', 'apple', 'apricot tart', 'cheese crêpe', 'meatloaf', 'peas', 'mashed potatoes', 'soggy bread without crust', 'big square plate', 'oversoaked polenta', 'soup of the day mushroom cream', 'cognac sauce', 'grated cheese', 'oversoaked chickpea curry', 'oversoaked roast beef', 'oversoaked food in pyramid shape', 'swiss chard vegetable ragout', 'oversoaked chia pudding', 'oversoaked mixed roast beef', 'cheese sauce', ' swiss chard', 'oversoaked mixed chickpea curry', 'sardinian fregola', 'smoked sausage (landjaeger)', 'soup of the day leek cream', 'vegetables for fregola', 'mustard', 'radish salad', 'fresh cheese praline', 'pickled cucumber', 'soup of the day banana-coconut', 'bean cassoulet', 'salami', 'pickled vegetables', 'deli meat cheese', 'turkey', 'sour cream', 'cylindrical transparent shot-glass', 'ricotta tortellini', 'potato vegetable curry', 'soup of the day sweetcorn', 'baked chickpea', 'tomato sauce', 'milk coffee', 'cherry jam', 'spreadable cheese', 'coffee cup', 'coffee plate', 'coffee yogurt', 'tilster cheese', 'brie cheese', 'margarine', 'appenzeller cheese', 'paprika sauce', 'bramata', 'green spelt dumplings', 'vegetable ragout for green spelt dumplings', 'chicken thigh steak', 'country cuts', 'veggie crispy bites', 'soup of the day tomatoes', 'vegetable salad for ham', 'country smoked ham', 'lye rolls', 'antipasti vegetables', 'tagliatelle tomato pesto antipasti', 'soup of the day beetroot', 'soy yogurt dip', 'vegan meatballs', 'vegetables for meatballs', 'yellow pea puree', 'boiled beef', 'horseradish bouillon', 'beef lasagna', 'eggplant cordon bleu', 'saffron risotto', 'bell pepper stew', 'pineapple-quark-mousse', 'quinoa salad', 'dried tomatoes', 'endives orange salad', 'orange fillet', 'cashew nuts', 'mascarpone', 'shiitake', 'red onion', 'risotto', 'vegetable salad', 'quinoa', 'little glass bowl', 'vegetable salad for quinoa', 'thai glass noodle salad', 'cheese ravioli', 'fruit quark', 'halibut', 'hummus', 'vegetables for halibut', 'asian dip', 'potato hash brown (roesti) with vegetables', 'oversoaked salmon fillet', 'oversoaked chickpea puree', 'bell pepper', 'port wine pears rucola risotto', 'soup of the day barley', 'gorgonzola', 'creamy polenta medium', 'beef patties in juicy sauce', 'merlot sauce', 'oversoaked perch fillet', 'oversoft food in crescent shape', 'rocket risotto', 'oversoaked bell peppers', 'oversoakeboiled beef', 'oversoakeboiled polenta', 'oversoaked carrots', 'polenta', 'soft zuchetti', 'potato herb patties', 'veggie cervalat sausage', 'colorful vegetables for veggie cervalat sausage', 'grisons barley soup', 'carrot puree', 'cod', 'ratatouille', 'soup of the day cauliflower cream', 'herbal semolina slice', 'crispy vegetable roll', 'rice noodle salad', 'soup of the day parmesan foam', 'creamed spinach', 'gnocchi pan tofu', 'homemade fishburgers', 'black bean puree', 'soup of the day zucchetti', 'roast beef', 'pizokel vegetable gratin', 'egg vinaigrette', 'cheesy soft egg noodles (kaesespaetzle)', 'cabbage salad', 'fennel salad for bowl', 'tree nut dressing', 'spelt marinated', 'feta marinated', 'beetroot cooked', 'iceberg lettuce', 'oversoaked smoked salmon', 'softened panna cotta', 'protein bowl', 'oversoaked fennel', 'oversoaked couscous', 'oversoaked turkey plate', 'lemon roulade', 'wild rice raw', 'homemade veggie burger', 'spicy tomato vegetable sauce', 'champignon organic', 'pork steak', 'gruyère', 'swiss macaroni and cheese', 'cantadou cheese', 'white bean puree', 'tilapia', 'burrito', 'pernod sauce', 'diced tomatoes', 'sliced beef', 'knot rolls', 'spelt goulash', 'french dressing', 'beans', 'salad', 'minced poultry patties', 'tofu', 'carbonara', 'carbonara tofu', 'rosemary sauce')


def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class GroundingDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        #Set this variable equal to True here if you would like to get logs for debugging
        self.logging_enabled = True
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict


    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        ####################### custom #########################
        do_closed_set_training = True

        # Run this code when creating the annotations and add 'tokens_positive' to each annotation
        aug_text_prompts = [ALL_LABELS]
        aug_label_list = aug_text_prompts[0]
#         # Split aug_label_list into chunks that fit within the model's max token limit
#         # ToDo: do not hardcode factor of 5, but count number of tokens
#         chunked_aug_label_list = chunks(aug_label_list, self.language_model.max_tokens // 5)

#         aug_tokens_positive = []
#         for chunk in chunked_aug_label_list:
#             _, _, tokens_positive, _ = self.get_tokens_and_prompts(chunk, True)
#             aug_tokens_positive.extend(tokens_positive)
        #######################################################

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, entities = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    if do_closed_set_training:
                        if self.logging_enabled:
                            print("gt_label: ", gt_label)
                            print("tokens_positive: ", tokens_positive)
                            print("text_prompts: ", text_prompts)
                        new_tokens_positive = [
                            tokens_positive[label] for label in gt_label
                        ]
                        _, positive_map = self.get_positive_map(
                            tokenized, new_tokens_positive)
                        positive_maps.append(positive_map)
                    else:  # open-set training
                        new_tokens_positive = self.get_tokens_positive_from_prompt(
                           gt_label, entities, tokens_positive, aug_label_list)
                        if self.logging_enabled:
                            print(f"caption_string: {caption_string}")
                            print(f"gt_label: {gt_label}")
                            print(f"new_tokens_positive: {new_tokens_positive}")
                        _, positive_map = self.get_positive_map(
                            tokenized, new_tokens_positive)
                        if self.logging_enabled:
                            print(f"positive_map: {positive_map}")
                        positive_maps.append(positive_map)

            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, entities = \
                        self.get_tokens_and_prompts(
                            text_prompt, True) 
                    new_text_prompts.append(caption_string)
                    if do_closed_set_training:
                        new_tokens_positive = [
                            tokens_positive[label] for label in gt_label
                        ]
                        _, positive_map = self.get_positive_map(
                            tokenized, new_tokens_positive)
                        positive_maps.append(positive_map)
                    else:  # open-set training
                        new_tokens_positive = self.get_tokens_positive_from_prompt(gt_label, entities, tokens_positive, aug_label_list)
                        if self.logging_enabled:
                            print(f"caption_string: {caption_string}")
                            print(f"gt_label: {gt_label}")
                            print(f"new_tokens_positive: {new_tokens_positive}")
                        _, positive_map = self.get_positive_map(
                            tokenized, new_tokens_positive)
                        if self.logging_enabled:
                            print(f"positive_map: {positive_map}")
                        positive_maps.append(positive_map)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses
    
    def get_tokens_positive_from_prompt(self, gt_label, entities, tokens_positive, aug_label_list):
        if self.logging_enabled:
            print(f"Entering get_tokens_positive_from_prompt with gt_label: {gt_label}, entities: {entities}, aug_label_list: {aug_label_list}")
            print(f"tokens_positive: {tokens_positive}")

        new_tokens_positive = []
        for label in gt_label:
            raw_label_text = aug_label_list[label.item()]  # Get the corresponding string from the augmented label list         
            label_text = clean_label_name(raw_label_text)  # Clean the label using the clean_label_name method         
            if self.logging_enabled:             
                print(f"Processing label: {label}, raw_label_text: {raw_label_text}, cleaned_label_text: {label_text}")         
            for idx, word in enumerate(entities):
                if self.logging_enabled:
                    print(f"Checking word: {word}, idx: {idx}")
                if word == label_text:
                    new_tokens_positive.append(tokens_positive[idx])
                    break

        if self.logging_enabled:
            print(f"Exiting get_tokens_positive_from_prompt with new_tokens_positive: {new_tokens_positive}")
        return new_tokens_positive





    import warnings
    import copy
    
    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        print("🔍 Debugging `batch_data_samples` structure:")
        for idx, data_sample in enumerate(batch_data_samples):
            print(f"\n📌 Sample {idx}: {data_sample}")
            print(f"✅ Available keys: {dir(data_sample)}")  # List all attributes in the object
    
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
    
        for data_samples in batch_data_samples:
            # Print raw data sample before processing
            print(f"\n🔹 Processing data sample: {data_samples}")
    
            # Extract `text` safely
            if hasattr(data_samples, "text"):
                text_prompts.append(data_samples.text)
                print(f"📝 Extracted text: {data_samples.text}")
            else:
                text_prompts.append(None)
                print("⚠️ Warning: `text` attribute is missing.")
    
            # Extract `caption_prompt` safely
            if hasattr(data_samples, "caption_prompt"):
                enhanced_text_prompts.append(data_samples.caption_prompt)
                print(f"📢 Extracted caption prompt: {data_samples.caption_prompt}")
            else:
                enhanced_text_prompts.append(None)
                print("⚠️ Warning: `caption_prompt` attribute is missing.")
    
            # Extract `tokens_positive` safely
            if hasattr(data_samples, "tokens_positive"):
                tokens_positives.append(data_samples.tokens_positive)
                print(f"🟢 Extracted tokens_positive: {data_samples.tokens_positive}")
            else:
                tokens_positives.append(None)
                print("⚠️ Warning: `tokens_positive` attribute is missing.")
    
        # Extract `custom_entities` flag
        if hasattr(batch_data_samples[0], "custom_entities"):
            custom_entities = batch_data_samples[0].custom_entities
            print(f"🔖 Custom entities detected: {custom_entities}")
        else:
            custom_entities = False
            print("⚠️ Warning: `custom_entities` attribute is missing.")
    
        # Process text prompts and token positives
        if len(text_prompts) == 1:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0], tokens_positives[0]
                )
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompt, custom_entities, enhanced_text_prompt, tokens_positive
                )
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives
                )
            ]
    
        token_positive_maps, text_prompts, _, entities = zip(*_positive_maps_and_prompts)
    
        print(f"\n✅ Processed token_positive_maps: {token_positive_maps}")
        print(f"✅ Processed text_prompts: {text_prompts}")
        print(f"✅ Processed entities: {entities}")
    
        # Extract image features
        visual_feats = self.extract_feat(batch_inputs)
    
        if isinstance(text_prompts[0], list):
            assert len(batch_inputs) == 1, "❌ Error: Only batch size 1 is supported for chunked text prompts."
            results_list = []
            count = 0
            entities = [[item for lst in entities[0] for item in lst]]
    
            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
    
                if self.text_feat_map is not None:
                    text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])
    
                batch_data_samples[0].token_positive_map = token_positive_maps_once
    
                head_inputs_dict = self.forward_transformer(copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples
                )[0]
    
                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
    
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            text_dict = self.language_model(list(text_prompts))
    
            if self.text_feat_map is not None:
                text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])
    
            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]
    
            head_inputs_dict = self.forward_transformer(visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples
            )
    
        for data_sample, pred_instances, entity, is_rec_task in zip(
            batch_data_samples, results_list, entities, is_rec_tasks
        ):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            "❌ Unexpected output: Possible Named Entity Recognition issue. "
                            "Try setting `custom_entities=True` and re-running."
                        )
                        label_names.append("unobject")
                    else:
                        label_names.append(entity[labels])
    
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
    
        print("✅ Successfully completed prediction and assigned results to batch data samples.")
        return batch_data_samples

