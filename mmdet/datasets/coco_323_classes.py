# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class Coco323Dataset(BaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes': (
        'beetroot-steamed-without-addition-of-salt',
        'bread_wholemeal',
        'jam',
        'water',
        'bread',
        'banana',
        'soft_cheese',
        'ham_raw',
        'hard_cheese',
        'cottage_cheese',
        'coffee',
        'fruit_mixed',
        'pancake',
        'tea',
        'salmon_smoked',
        'avocado',
        'spring_onion_scallion',
        'ristretto_with_caffeine',
        'ham_n_s',
        'egg',
        'bacon',
        'chips_french_fries',
        'juice_apple',
        'chicken',
        'tomato',
        'broccoli',
        'shrimp_prawn',
        'carrot',
        'chickpeas',
        'french_salad_dressing',
        'pasta_hornli_ch',
        'sauce_cream',
        'pasta_n_s',
        'tomato_sauce',
        'cheese_n_s',
        'pear',
        'cashew_nut',
        'almonds',
        'lentil_n_s',
        'mixed_vegetables',
        'peanut_butter',
        'apple',
        'blueberries',
        'cucumber',
        'yogurt',
        'butter',
        'mayonnaise',
        'soup',
        'wine_red',
        'wine_white',
        'green_bean_steamed_without_addition_of_salt',
        'sausage',
        'pizza_margherita_baked',
        'salami_ch',
        'mushroom',
        'tart_n_s',
        'rice',
        'white_coffee',
        'sunflower_seeds',
        'bell_pepper_red_raw',
        'zucchini',
        'asparagus',
        'tartar_sauce',
        'lye_pretzel_soft',
        'cucumber_pickled_ch',
        'curry_vegetarian',
        'soup_of_lentils_dahl_dhal',
        'salmon',
        'salt_cake_ch_vegetables_filled',
        'orange',
        'pasta_noodles',
        'cream_double_cream_heavy_cream_45',
        'cake_chocolate',
        'pasta_spaghetti',
        'black_olives',
        'parmesan',
        'spaetzle',
        'salad_lambs_ear',
        'salad_leaf_salad_green',
        'potato',
        'white_cabbage',
        'halloumi',
        'beetroot_raw',
        'bread_grain',
        'applesauce',
        'cheese_for_raclette_ch',
        'bread_white',
        'curds_natural',
        'quiche',
        'beef_n_s',
        'taboule_prepared_with_couscous',
        'aubergine_eggplant',
        'mozzarella',
        'pasta_penne',
        'lasagne_vegetable_prepared',
        'mandarine',
        'kiwi',
        'french_beans',
        'spring_roll_fried',
        'caprese_salad_tomato_mozzarella',
        'leaf_spinach',
        'roll_of_half_white_or_white_flour_with_large_void',
        'omelette_with_flour_thick_crepe_plain',
        'tuna',
        'dark_chocolate',
        'sauce_savoury_n_s',
        'raisins_dried',
        'ice_tea_on_black_tea_basis',
        'kaki',
        'smoothie',
        'crepe_with_flour_plain',
        'nuggets',
        'chili_con_carne_prepared',
        'veggie_burger',
        'chinese_cabbage',
        'hamburger',
        'soup_pumpkin',
        'sushi',
        'chestnuts_ch',
        'sauce_soya',
        'balsamic_salad_dressing',
        'pasta_twist',
        'bolognaise_sauce',
        'leek',
        'fajita_bread_only',
        'potato_gnocchi',
        'rice_noodles_vermicelli',
        'bread_whole_wheat',
        'onion',
        'garlic',
        'hummus',
        'pizza_with_vegetables_baked',
        'beer',
        'glucose_drink_50g',
        'ratatouille',
        'peanut',
        'cauliflower',
        'green_olives',
        'bread_pita',
        'pasta_wholemeal',
        'sauce_pesto',
        'couscous',
        'sauce',
        'bread_toast',
        'water_with_lemon_juice',
        'espresso',
        'egg_scrambled',
        'juice_orange',
        'braided_white_loaf_ch',
        'emmental_cheese_ch',
        'hazelnut_chocolate_spread_nutella_ovomaltine_caotina',
        'tomme_ch',
        'hazelnut',
        'peach',
        'figs',
        'mashed_potatoes_prepared_with_full_fat_milk_with_butter',
        'pumpkin',
        'swiss_chard',
        'red_cabbage_raw',
        'spinach_raw',
        'chicken_curry_cream_coconut_milk_curry_spices_paste',
        'crunch_muesli',
        'biscuit',
        'meatloaf_ch',
        'fresh_cheese_n_s',
        'honey',
        'vegetable_mix_peas_and_carrots',
        'parsley',
        'brownie',
        'ice_cream_n_s',
        'salad_dressing',
        'dried_meat_n_s',
        'chicken_breast',
        'mixed_salad_chopped_without_sauce',
        'feta',
        'praline_n_s',
        'walnut',
        'potato_salad',
        'kolhrabi',
        'alfa_sprouts',
        'brussel_sprouts',
        'gruyere_ch',
        'bulgur',
        'grapes',
        'chocolate_egg_small',
        'cappuccino',
        'crisp_bread',
        'bread_black',
        'rosti_n_s',
        'mango',
        'muesli_dry',
        'spinach',
        'fish_n_s',
        'risotto',
        'crisps_ch',
        'pork_n_s',
        'pomegranate',
        'sweet_corn',
        'flakes',
        'greek_salad',
        'sesame_seeds',
        'bouillon',
        'baked_potato',
        'fennel',
        'meat_n_s',
        'croutons',
        'bell_pepper_red_stewed',
        'nuts',
        'breadcrumbs_unspiced',
        'fondue',
        'sauce_mushroom',
        'strawberries',
        'pie_plum_baked_with_cake_dough',
        'potatoes_au_gratin_dauphinois_prepared',
        'capers',
        'bread_wholemeal_toast',
        'red_radish',
        'fruit_tart',
        'beans_kidney',
        'sauerkraut',
        'mustard',
        'country_fries',
        'ketchup',
        'pasta_linguini_parpadelle_tagliatelle',
        'chicken_cut_into_stripes_only_meat',
        'cookies',
        'sun_dried_tomatoe',
        'bread_ticino_ch',
        'semi_hard_cheese',
        'porridge_prepared_with_partially_skimmed_milk',
        'juice',
        'chocolate_milk',
        'bread_fruit',
        'corn',
        'dates',
        'pistachio',
        'cream_cheese_n_s',
        'bread_rye',
        'witloof_chicory',
        'goat_cheese_soft',
        'grapefruit_pomelo',
        'blue_mould_cheese',
        'guacamole',
        'tofu',
        'cordon_bleu',
        'quinoa',
        'kefir_drink',
        'salad_rocket',
        'pizza_with_ham_with_mushrooms_baked',
        'fruit_coulis',
        'plums',
        'pizza_with_ham_baked',
        'pineapple',
        'seeds_n_s',
        'focaccia',
        'mixed_milk_beverage',
        'coleslaw_chopped_without_sauce',
        'sweet_potato',
        'chicken_leg',
        'croissant',
        'cheesecake',
        'sauce_cocktail',
        'croissant_with_chocolate_filling',
        'pumpkin_seeds',
        'artichoke',
        'soft_drink_with_a_taste',
        'apple_pie',
        'white_bread_with_butter_eggs_and_milk',
        'savoury_pastry_stick',
        'tuna_in_oil_drained',
        'meat_terrine_pate',
        'falafel_balls',
        'berries_n_s',
        'latte_macchiato',
        'sugar_melon_galia_honeydew_cantaloupe',
        'mixed_seeds_n_s',
        'oil_vinegar_salad_dressing',
        'celeriac',
        'chocolate_mousse',
        'lemon',
        'chocolate_cookies',
        'birchermuesli_prepared_no_sugar_added',
        'muffin',
        'pine_nuts',
        'french_pizza_from_alsace_baked',
        'chocolate_n_s',
        'grits_polenta_maize_flour',
        'wine_rose',
        'cola_based_drink',
        'raspberries',
        'roll_with_pieces_of_chocolate',
        'cake_lemon',
        'rice_wild',
        'gluten_free_bread',
        'pearl_onion',
        'tzatziki',
        'ham_croissant_ch',
        'corn_crisps',
        'lentils_green_du_puy_du_berry',
        'rice_whole_grain',
        'cervelat_ch',
        'aperitif_with_alcohol_n_s_aperol_spritz',
        'peas',
        'tiramisu',
        'apricots',
        'lasagne_meat_prepared',
        'brioche',
        'vegetable_au_gratin_baked',
        'basil',
        'butter_spread_puree_almond',
        'pie_apricot',
        'rusk_wholemeal',
        'pasta_in_conch_form',
        'pasta_in_butterfly_form_farfalle',
        'damson_plum',
        'shoots_n_s',
        'coconut',
        'banana_cake',
        'sauce_curry',
        'watermelon_fresh',
        'white_asparagus',
        'cherries',
        'nectarine',)
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
