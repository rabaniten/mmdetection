# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


ALL_CLASSES = ('chervil cream sauce', 'small sauce-glass', 'cream tart', 'white plate', 'chicken breast', 'tortellini', 'shallow bowl', 'cherry tomato', 'mixed dish of tortellini and chervil cream sauce', 'caesar salad', 'parmesan dressing', 'egg cooked', 'toast croutons', 'parmesan shavings', 'bacon', 'lettuce', 'mixed dish of caesar salad and dressing', 'colorful vegetable pan with soft egg noodles (spaetzle)', 'white plate without rim', 'capuns', 'soup of the day ratatouille cream', 'root vegetables', 'soup bowl', 'beer sauce', 'finger-shaped potato dumplings (schupfnudeln)', 'sauerkraut', 'smoked pork neck', 'roasted cashew nuts', 'hioumi', 'raspberry quark', 'natural yogurt', 'strawberry yogurt', 'apricot yogurt', 'raspberry yogurt', 'beans green', 'cream sauce', 'vegetable bolognese', 'potatoes', 'mixed dish of horn-shaped pasta (hoernli) and vegetable bolognese', 'caramel flan', 'large sauce-glass', 'fruit salad', 'soup of the day artichoke', 'veal sausage', 'onion sauce', 'small glass fruitsalad bowl', 'spanish tortilla with basil sauce', 'vanilla cream puffs', 'small quadratic plate-bowl', 'spanish tortilla', 'quadratic dessert-plate', 'horn-shaped pasta (hoernli)', 'radish', 'sausage and cheese salad', 'lollo bianco', 'house bread', 'soup of the day potato', 'vegetarian bami goreng', 'lollo rosso', 'soup of the day broccoli cream', 'mixed dish of spaghetti vegetable strips and saffron herb sauce', 'vegetable strips saffron-herb sauce', 'pureed green balls', 'spaghetti', 'pureed food in a special shape', 'eggplant moussaka', 'pureed meat slices', 'pureed food in oval shape', 'pureed broccoli', 'pureed mashed potatoes', 'spinach tart', 'turmeric sauce', 'pita bread', 'mixed dish of trout fillet spinach rice and sauce', 'sprout vegetables', 'scallion', 'herbal rice', 'soup of the day curry cream', 'lentil ragout', 'spinach', 'trout fillet', 'mixed meal of lentil ragout paneer pita and sprouts', 'quinoa patties', 'vegetables for quinoa patties', 'toast', 'filling of toast hawaii', 'onion red', 'cranberry', 'barley risotto', 'salad leaves', 'cheese tart', 'zucchetti', 'lemon panna cotta', 'parsley fritters', 'lemon', 'capers', 'smoked trout', 'trout tartare', 'soup of the day yellow pea', 'horseradish foam', 'butter', 'pesto sauce', 'vegetable lasagna', 'applesauce', 'broccoli', 'dill mashed potatoes', 'salmon cubes marinated', 'white wine sauce', 'brown sauce', 'pureed food in pyramid shape', 'penne rigate', 'pureed polenta', 'pureed balls', 'tagliatelle', 'pureed cauliflower', 'raspberry', 'raspberry mousse in pyramid shape', 'bolognese', 'white sauce', 'bell pepper sauce', 'chocolate mousse', 'round raspberry mousse', 'romanesco', 'slices', 'poultry stew', 'pureed salmon', 'pureed chicken thigh', 'pureed fries', 'pureed sausage', 'mixed dish of penne and bolognese', 'plate with red rim', 'mixed meal of soft egg noodles (spaetzle) and halloumi', 'veggie swiss macaroni and cheese', 'mixed dish of chicken bell pepper sauce romanesco and pasta', 'boiled meat salad seed oil', 'vegetable strudel', 'vegetable patch', 'herb quark dip', 'vegetables for boiled meat salad', 'plum tart', 'boiled meat', 'mixed meal of vegetable strudel and herb quark dip', 'gnocchi seitan pan', 'mixed meal of lamb stew bulgur and carrots', 'carrots', 'currant sheet cake', 'lamb stew', 'sauce', 'bulgur', 'mixed dish of gnocchi seitan pan and oyster mushrooms', 'vegetable salad with white beans', 'vegetables for green spelt risotto', 'green spelt risotto', 'rye bread', 'apple tart', 'bouillon', 'cold chicken breast', 'cocktail sauce', 'soup of the day carrot cream', 'curry dip', 'soup of the day lentil ginger', 'poulet cordon bleu', 'sliced seitan', 'mixed dish of poulet cordon bleu cauliflower soft egg noodles (spaetzle) and jus', 'mixed dish of sliced seitan pilaf rice and sugar peas', 'roasted cauliflower', 'pilau rice', 'soft egg noodles (spaetzle)', 'jus', 'sugar peas', 'sauce for sliced seitan', 'boiled potatoes', 'semolina porridge', 'cherry compote', 'cinnamon sugar', 'small plastic cup', 'oversoaked sliced chicken', 'mixed meal of semolina porridge and cherry compote', 'overly soft thick brie cheese', 'overly soft cottage cheese', 'mixed meal of meat cheese, mustard sauce and lyonnaise potatoes', 'oversoaked sliced veal', 'overly soft cream cheese', 'overly soft thin brie cheese', 'currants', 'lyonnaise potatoes', 'meat cheese', 'mustard sauce', 'soup of the day bell peppers', 'sliced quorn sauce zurich style', 'sliced quorn zurich style', 'colorful vegetables from zuchetti peas carrots and beans', 'hash brown (roesti)', 'bread dumplings', 'sauce poultry ragout', 'mixed dish of poultry ragout and bread dumplings', 'mixed meal of quorn vegetables and hash brown (roesti)', 'banana organic', 'croissant', 'lye croissant', 'coffee', 'jug lid on the ground', 'uncovered jug', 'jug covered with lid', 'orange juice', 'multigrain roll', 'scrambled eggs', 'milk', 'mueesli', 'large glass fruitsalad-bowl', 'milk roll', 'mozzarella', 'baked vegetables for mozzarella', 'wedges', 'chipolata sausage', 'vegetables for chipolata sausage', 'rocket', 'walnut', 'mixed meal of mozzarella salad, baked vegetables and house bread', 'vegetable piccata made from zuchetti eggplant and piccata mass', 'bramata slice', 'vegetables for piccata', 'cream', 'chocolate cake', 'fried rice', 'beef meatballs', 'spicy vegetable ragout', 'olives', 'cucumber', 'tomato', 'turkey breast', 'carrot', 'chili with vegetables', 'lenses brown', 'mixed dish of chili with vegetables lentils and romanesco', 'pear', 'apple', 'apricot tart', 'cheese crêpe', 'mixed meal of meatloaf peas mashed potatoes and sauce', 'soggy bread without crust', 'large square plate', 'oversoaked polenta', 'meatloaf', 'mashed potatoes', 'peas', 'soup of the day mushroom cream', 'cognac sauce', 'grated cheese', 'oversoaked chickpea curry', 'rice', 'oversoaked roast beef', 'oversoaked food in pyramid shape', 'swiss chard vegetable ragout', 'oversoaked chia pudding', 'oversoaked mixed roast beef', 'mixed dish of cream cheese sauce and swiss chard', 'oversoaked mixed chickpea curry', 'sardinian fregola', 'smoked sausage (landjaeger)', 'soup of the day leek cream', 'sardinian fregola with vegetables', 'vegetables for fregola', 'mustard', 'radish salad', 'fresh cheese praline', 'pickled cucumber', 'sliced quorn', 'soup of the day banana-coconut', 'bean cassoulet', 'salami', 'pickled vegetables', 'deli meat cheese', 'turkey', 'sour cream', 'cylindrical transparent shot glass', 'turkey ham', 'ricotta tortellini', 'potato vegetable curry', 'soup of the day sweetcorn', 'baked chickpea', 'cream sauce and tomato sauce', 'milk coffee', 'cherry jam', 'gruyere cheese', 'spreadable cheese', 'coffee cup', 'coffee plate', 'coffee yogurt', 'tilster cheese', 'brie cheese', 'margarine', 'appenzeller cheese', 'paprika sauce', 'bramata', 'green spelt dumplings', 'vegetable ragout for green spelt dumplings', 'chicken thigh steak', 'country cuts', 'veggie crispy bites', 'soup of the day tomatoes', 'vegetable salad for ham', 'country smoked ham', 'lye rolls', 'antipasti vegetables', 'tagliatelle tomato pesto antipasti', 'soup of the day beetroot', 'soy yogurt dip', 'vegan meatballs', 'vegetables for meatballs', 'yellow pea puree', 'boiled beef', 'horseradish bouillon', 'beef lasagna', 'eggplant cordon bleu', 'saffron risotto', 'bell pepper stew', 'pineapple-quark-mousse', 'tomato sauce', 'quinoa salad', 'dried tomatoes', 'endives orange salad', 'orange fillet', 'cashew nuts', 'creamy risotto', 'red onion', 'risotto', 'vegetable salad with quinoa', 'small glass bowl', 'vegetable salad for quinoa', 'thai glass noodle salad', 'cheese ravioli', 'fruit quark', 'pineapple', 'halibut', 'hummus', 'vegetables for halibut', 'asian dip', 'potato hash brown (roesti) with vegetables', 'oversoaked salmon fillet', 'oversoaked chickpea puree', 'bell pepper', 'port wine pears rucola risotto', 'soup of the day barley', 'rocket risotto', 'creamy polenta medium', 'beef patties in juicy sauce', 'merlot sauce', 'oversoaked perch fillet', 'oversoft food in crescent shape', 'mixed dish of rucola risotto gorgonzola and walnut', 'oversoaked mixed dish of salmon fillet, chickpea puree and bell peppers', 'oversoaked mixed dish of boiled beef polenta and carrots', 'mixed dish of beef patties in juicy sauce broccoli and polenta', 'oversoaked mixed dish of crescent-shaped perch fillets and zuchetti', 'potato herb patties', 'veggie cervalat sausage', 'colorful vegetables for veggie cervalat sausage', 'grisons barley soup', 'carrot puree', 'cod', 'ratatouille', 'soup of the day cauliflower cream', 'herbal semolina slice', 'crispy vegetable roll', 'rice noodle salad', 'soup of the day parmesan foam', 'creamed spinach', 'gnocchi pan tofu', 'homemade fishburgers', 'black bean puree', 'soup of the day zucchetti', 'roast beef', 'pizokel vegetable gratin', 'egg vinaigrette', 'cheesy soft egg noodles (kaesespaetzle)', 'cabbage salad', 'fennel salad for bowl', 'tree nut dressing', 'spelt marinated', 'feta marinated', 'beetroot cooked', 'iceberg lettuce', 'oversoaked smoked salmon', 'oversoaked mixed dish of smoked salmon zuchetti and panna cotta', 'mixed dish of protein bowl and multigrain roll', 'oversoaked fennel', 'oversoaked couscous', 'oversoaked turkey plate', 'lemon roulade', 'wild rice raw', 'homemade veggie burger', 'spicy tomato vegetable sauce', 'champignon organic', 'pork steak', 'gruyère', 'swiss macaroni and cheese', 'cantadou cheese', 'white bean puree', 'tilapia', 'burrito', 'pernod sauce', 'diced tomatoes', 'sliced beef', 'knot rolls', 'spelt goulash', 'french dressing', 'vegetables for salade nicoise consisting of beans spinach and lettuce', 'minced poultry patties', 'mixed dish of penne tofu and carbonara', 'carbonara tofu', 'rosemary sauce')

@DATASETS.register_module()
class CocoDataset(BaseDetDataset):
    """Dataset for COCO."""
    
    # custom
    # ToDo: set as class variable...?
    do_closed_set_training = False
    print('\nclosed_set_training\n' if do_closed_set_training else '\nopen set training\n')

    if do_closed_set_training:
        METAINFO = {
            'classes': ALL_CLASSES
        }
    else:  # provide, extra wrong classes for open set training
        METAINFO = {
            'classes': ALL_CLASSES
        }
    
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True
            
    def print_gt_labels(self, data_info: dict):
        """
        Print image file name together with all GT labels for the image.
        Print a warning if no GT labels are loaded.
        
        Args:
            data_info (dict): A dictionary containing image data and annotations.
        """
        img_path = data_info.get('img_path', 'Unknown Image Path')
        instances = data_info.get('instances', [])

        # Collect all labels for the current image
        gt_labels = [instance['bbox_label'] for instance in instances if 'bbox_label' in instance]

        # If no labels are found, print a warning
        if not gt_labels:
            print(f"Warning: No GT labels found for image {img_path}")
        else:
            print(f"Image: {img_path}")
            print(f"GT Labels: {gt_labels}")
            print('-' * 50)
            
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
                raw_img_info,
                'closed_set_training':
                self.do_closed_set_training,
            })
            data_list.append(parsed_data_info)
            # Print image file name and ground truth labels
            self.print_gt_labels(parsed_data_info)
            
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list


    
    #ToDo: maybe use self.do_closed_set_training instead of getting it in the input...?
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        do_closed_set_training = raw_data_info['closed_set_training']  # new

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
            if do_closed_set_training:
                data_info['text'] = self.metainfo['classes']  # closed-set predictions by defaut
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True
            
            
        #ToDo: Clean up copied code for closed set and open set case    
        # open-set training
        if not do_closed_set_training:
            instances = []
            labels_for_text_input = []
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
                
                # new: save image-specific class labels
                cat_name = self.coco.loadCats(ann['category_id'])[0]["name"]
                labels_for_text_input.append(cat_name)
            
            # new: save image-specific class labels in appropriate format
            unique_labels_for_text_input = set(labels_for_text_input)
            
            #print("Debugging in coco.py: img_path", img_path)
            #print("Debugging in coco.py: img_id", img_info['img_id'])
            print("Debugging in coco.py: unique_labels_for_text_input", unique_labels_for_text_input)
            
            data_info['text'] = tuple(unique_labels_for_text_input)
            #print('image-specific input text:', data_info['text'])
            #print('data format:', type(data_info['text']))
            #print('default data format:', type(self.metainfo['classes']))

        # closed-set training (default)
        else:
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
