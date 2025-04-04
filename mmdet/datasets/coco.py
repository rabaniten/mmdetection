# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset

#ToDo: remove labels here, they can be set dynamically in the config


ALL_LABELS = (
    'Other',
    'banana',
    'caramel flan',
    'lamb stew',
    'carrots',
    'bread',
    'soup',
    'chanterelle parsley risotto',
    'panna cotta',
    'french salad dressing',
    'corn',
    'pomegranate',
    'chocolate ice cream',
    'applesauce',
    'bulgur',
    'gnocchi seitan pan',
    'whipped cream',
    'Plum muffin',
    'grated cheese',
    'lettuce',
    'mashed potatoes',
    'mashed potato',
    'oversoaked beef roast',
    'peas',
    'carrot',
    'chipolata',
    'potato wedges',
    'italian sauce',
    'turnip cabbage',
    'vanilla ice cream',
    'apple',
    'fruit salad',
    'mozzarella salad',
    'zucchini',
    'pureed carrot',
    'pureed beef',
    'cream sauce',
    'lemon slice',
    'caper butter sauce',
    'potatoes',
    'sole',
    'beans',
    'cherry tomato',
    'mixed salad',
    'meatloaf',
    'cognac sauce',
    'herb cream sauce',
    'eggplant moussaka',
    'strawberry yogurt',
    'hollandaise sauce',
    'legume salad',
    'macaroni',
    'tomato sauce',
    'vanilla porridge',
    'apricot yogurt',
    'cream slice',
    'butter',
    'gravy',
    'bircher muesli',
    'boiled egg',
    'Cherry Tomato',
    'vegetables for boiled meat salad',
    'shrimps',
    'walnut',
    'cheese spaetzle',
    'cabbage salad',
    'baby lettuce',
    'radish',
    'orange',
    'cocktail sauce',
    'parsley',
    'fried onions',
    'vegetable strudel',
    'herb quark dip',
    'chocolate yogurt',
    'mixed carrot and peas',
    'rice',
    'coffee yogurt',
    'pureed bratwurst',
    'vanilla cream',
    'herb rice',
    'trout fillet',
    'spinach leaf',
    'thin chocolate decoration',
    'salmon cubes',
    'oversoaked millet slice',
    'zucchetti',
    'oversoaked broccoli',
    'oversoaked sliced chicken',
    'oversoaked bean',
    'rustico croissant',
    'multigrain roll',
    'honey',
    'protein drink',
    'cheese crepe',
    'swiss chard vegetable ragout',
    'mustard sauce',
    'mint',
    'moustard sauce',
    'baked meatloaf',
    'lingonberry compote',
    'Quail Breast',
    'Cranberries',
    'Celery Salad',
    'jam',
    'bread roll',
    'croissant',
    'milk',
    'chocolate powder bag',
    'sliced veal',
    'romanesco',
    'oversoaked zucchetti',
    'polenta',
    'broccoli',
    'beef braised slice',
    'tiramisu slice',
    'merlot jus',
    'thin chocolate',
    'balsamic sauce',
    'strawberry ice cream',
    'juniper jus',
    'veal cheak',
    'fregola',
    'red cabbage',
    'red chicory',
    'cooked beetroot',
    'orange segments',
    'walnuts',
    'peeled carrot',
    'iceberg lettuce',
    'walnut dressing',
    'chocolate mousse',
    'grapes',
    'cheese',
    'salad',
    'carrot appetizer',
    'tilapia',
    'pernod sauce',
    'white bean puree',
    'vegetable',
    'nut cake',
    'chopped herbs',
    'tiramisu',
    'boiled potatoes',
    'cheese spread',
    'minced meat',
    'oversoaked chicken strips with sauce',
    'penne pasta',
    'scrambled eggs',
    'creamed spinach',
    'rice noodle salad',
    'chili pepper',
    'peanuts',
    'lactose-free milk',
    'Jam sandwich cookie',
    'chicken breast',
    'yellow pea puree',
    'plain yogurt',
    'apple tart',
    'cream',
    'veggie crispy bites',
    'oversoaked polenta',
    'Pork Steak',
    'soft egg noodles (spaetzle)',
    'Root Vegetables',
    'vegetables for meatballs',
    'vegan meatballs',
    'Tagliatelle',
    'sauce',
    'vegetables',
    'cheese ravioli',
    'lemon sorbet',
    'cashew nuts',
    'smoked sausage',
    'mixed vegetables',
    'sardinian fregola',
    'sliced quorn',
    'basil sauce',
    'beef lasagna',
    'glazed carrots',
    'beef roast',
    'special bean',
    'saffron risotto',
    'bell pepper sauce',
    'eggplant cordon bleu',
    'peperonata',
    'oversoaked creamed spinach',
    'white wine',
    'cod',
    'spinach tart',
    'breaded poultry meatball',
    'rosemary sauce',
    'cauliflower',
    'yeast roll',
    'oversoaked bolognese',
    'chickpea puree',
    'bread without crust',
    'plum',
    'penne',
    'diced tomatoes',
    'carbonara tofu',
    'kiwi',
    'pizokel vegetable gratin',
    'plum tart',
    'rusk',
    'coffee',
    'emmental cheese',
    'margarine',
    'hash brown (roesti) with cheese',
    'orange segment',
    'oversoaked celery',
    'ham sandwich',
    'sweet potato',
    'soft cheese',
    'spreadable cheese',
    'fruit jelly',
    'quince jelly',
    'cottage cheese',
    'wild rice',
    'spicy tomato vegetable sauce',
    'vegetarian burger',
    'mushrooms',
    'sea bass',
    'bok choy',
    'oriental rice',
    'boiled beef',
    'root vegetables',
    'broth',
    'plum crumble',
    'mayonnaise',
    'greyerzer cheese',
    'apricose quark',
    'veggie sausage',
    'potato herb patties',
    'colorful vegetables for veggie cervalat sausage',
    'swedish cake',
    'golden berry',
    'oversoaked boiled potatoes',
    'chicken thigh',
    'bramata slice',
    'cream herb sauce',
    'vegetable ragout',
    'spelt dumplings',
    'brownie',
    'fruit quark',
    'gran padrone',
    'thai glass noodle salad',
    'knot bread rolls',
    'red wine',
    'leek',
    'colorful vegetable pan with soft egg noodles (spaetzle)',
    'Cashew Nuts',
    'Halloumi',
    'sauteed tomato',
    'tofu',
    'raspberry quark',
    'mac and cheese',
    'gruyere cheese',
    'coffee cream',
    'spinach',
    'sliced turkey breast',
    'tomato cream sauce',
    'tortellini',
    'oversoaked carrot',
    'beef tartare',
    'raspberry yogurt lactose-free',
    'quinoa',
    'vegetable salad',
    'apricot tart',
    'oversoaked salmon',
    'endive orange salad',
    'risotto',
    'fish burger',
    'black bean puree',
    'mashed black beans',
    'apple cookie',
    'pesto cream sauce',
    'gnocchi',
    'walnut pesto sauce',
    'oversoaked chickpea curry',
    'oversoaked poached salmon',
    'oversoaked bell pepper',
    'minced beef sauce',
    'pureed omelette',
    'egg',
    'spelt goulash',
    'cherry tomatoes',
    'wine',
    'black olives',
    'baby spinach',
    'beef meatballs',
    'fried rice',
    'vegetable stew',
    'eggplant piccata',
    'cheesecake',
    'vegetable piccata',
    'red onion',
    'spelt risotto',
    'dried tomato',
    'mashed pasta',
    'cold cuts',
    'vegetable lasagna',
    'white wine sauce',
    'salmon cube',
    'mashed potato with dill',
    'Radish',
    'Lollo Rosso',
    'Boiled Egg',
    'Sausage Cheese Salad',
    'Butter',
    'Bread',
    'bami goreng',
    'capuns with cheese',
    'grape',
    'sachertorte',
    'hawaiian toast',
    'Red Onion',
    'sour cream',
    'bell pepper',
    'goulash soup',
    'quinoa patty',
    'chicken cordon bleu',
    'veal steak',
    'duchess potatoes',
    'chili with vegetables',
    'veal sausage',
    'green beans',
    'vegetable bolognese',
    'radicchio rosso',
    'lemon',
    'toast bread',
    'smoked trout',
    'horseradish foam',
    'trout tartare',
    'capers',
    'parsley fritters',
    'mushroom risotto',
    'cheese tart',
    'chives',
    'bread dumpling',
    'poultry ragout',
    'sugar peas',
    'oversoaked scrambled egg',
    'potato',
    'scrambled egg',
    'sesame tofu',
    'rosti',
    'quorn strips in cream sauce',
    'oversoaked cauliflower',
    'pureed polenta',
    'apple juice',
    'cream tart',
    'chervil cream sauce',
    'croutons',
    'bacon',
    'salad dressing',
    'gran padano',
    'orange juice',
    'strawberry quark',
    'tilster cheese',
    'Chicken',
    'pasta',
    'cinnamon sugar',
    'cherry compote',
    'semolina porridge',
    'ketchup',
    'buttered pretzel',
    'beet ginger salad',
    'salmon',
    'cholocate drink',
    'lemon roulade',
    'pilaw rice',
    'burrito',
    'country cuts',
    'vegan  nuggets',
    'pilaf rice',
    'mashed semolina',
    'fruit',
    'Aufschnittteller VVG',
    'bean cassoulet',
    'salad leaves',
    'Choernlibroetli',
    'Aufschnittteller',
    'Bean Cassoulet',
    'cucumber',
    'salami',
    'turkey cold cut',
    'boiled eggs',
    'roast beef',
    'vinaigrette',
    'olives',
    'Halibut',
    'vegetables for halibut',
    'pasta with tomato sauce',
    'hummus',
    'whole grain rice cake',
    'tea',
    'horseradish bouillon',
    'salted potatoes',
    'root vegetable',
    'lamb loin',
    'special jus',
    'mustard',
    'bag of Ovaltine',
    'cheese sandwich',
    'cod with herbs',
    'ratatouille',
    'herb semolina slice',
    'carrot puree',
    'springroll',
    'halloumi',
    'roasted cashew nuts',
    'colorful spaetzle',
    'rye bread',
    'radishes',
    'brie cheese',
    'gruyere',
    'crispy fried onions',
    'chickpea triangles',
    'vegetable curry',
    'red pepperoncini',
    'spring onion',
    'oversoaked cheese plate',
    'celery',
    '"Salade nicoise"',
    'vegetable salad with white beans',
    'chicken breast slices',
    'curry sauce',
    'finger-shaped potato dumplings',
    'smoked pork neck',
    'sauerkraut',
    'pureed chicken',
    'Cranberry',
    'Seitan Strips',
    'Sauce',
    'Spanish Tortilla',
    'vegetable salad with feta',
    'mashed peas',
    'lentil ragout',
    'pita bread',
    'sprout vegetable',
    'muffin',
    'soft bircher muesli',
    'quark',
    'sweet and sour carrot',
    'salami sandwich',
    'mushroom cream sauce',
    'capuns',
    'small sauce-glass',
    'white plate',
    'shallow bowl',
    'caesar salad',
    'parmesan dressing',
    'egg cooked',
    'toast croutons',
    'parmesan shavings',
    'dressing',
    'white plate without rim',
    'soup of the day ratatouille cream',
    'soup-bowl',
    'beer sauce',
    'finger-shaped potato dumplings (schupfnudeln)',
    'hioumi',
    'natural yogurt',
    'raspberry yogurt',
    'beans green',
    'thyme',
    'horn-shaped pasta (hoernli)',
    'big sauce-glass',
    'soup of the day artichoke',
    'onion sauce',
    'glass fruitsalad-bowl',
    'spanish tortilla',
    'vanilla cream puffs',
    'small quadratic plate-bowl',
    'basil pesto',
    'quadratic dessert-plate',
    'sausage and cheese salad',
    'lollo bianco',
    'house bread',
    'soup of the day potato',
    'vegetarian bami goreng',
    'lollo rosso',
    'soup of the day broccoli cream',
    'spaghetti',
    'vegetable strips',
    'saffron herb sauce',
    'vegetable strips saffron-herb sauce',
    'pureed green balls',
    'pureed food in a special shape',
    'pureed meat slices',
    'pureed food in oval shape',
    'pureed broccoli',
    'pureed mashed potatoes',
    'turmeric',
    'sprout vegetables',
    'scallion',
    'herbal rice',
    'soup of the day curry cream',
    'paneer',
    'quinoa patties',
    'vegetables for quinoa patties',
    'toast',
    'turkey ham',
    'pineapple',
    'onion red',
    'cranberry',
    'barley risotto',
    'lemon panna cotta',
    'soup of the day yellow pea',
    'dill mashed potatoes',
    'salmon cubes marinated',
    'brown sauce',
    'pureed food in pyramid shape',
    'penne rigate',
    'pureed balls',
    'tagliatelle',
    'pureed cauliflower',
    'raspberry',
    'raspberry mousse in pyramid shape',
    'bolognese',
    'white sauce',
    'round raspberry mousse',
    'slices',
    'poultry stew',
    'pureed salmon',
    'pureed chicken thigh',
    'pureed fries',
    'pureed sausage',
    'plate with red rim',
    'veggie swiss macaroni and cheese',
    'poulet',
    'boiled meat salad seed oil',
    'vegetable patch',
    'boiled meat',
    'currant sheet cake',
    'bulgur sauce',
    'sliced seitan',
    'oyster mushrooms',
    'vegetables for green spelt risotto',
    'green spelt risotto',
    'bouillon',
    'cold chicken breast',
    'soup of the day carrot cream',
    'curry dip',
    'soup of the day lentil ginger',
    'poulet cordon bleu',
    'jus',
    'pilau rice',
    'roasted cauliflower',
    'sauce for sliced seitan',
    'small plastic cup',
    'overly soft thick brie cheese',
    'overly soft cottage cheese',
    'meat cheese',
    'lyonnaise potatoes',
    'oversoaked sliced veal',
    'overly soft cream cheese',
    'overly soft thin brie cheese',
    'currants',
    'soup of the day bell peppers',
    'sliced quorn sauce zurich style',
    'sliced quorn zurich style',
    'colorful vegetables from zuchetti peas carrots and beans',
    'hash brown (roesti)',
    'bread dumplings',
    'sauce poultry ragout',
    'banana organic',
    'lye croissant',
    'lid on the ground',
    'uncovered jug',
    'jug covered with lid',
    'mueesli',
    'large glass fruitsalad-bowl',
    'milk roll',
    'mozzarella',
    'baked vegetables for mozzarella',
    'wedges',
    'chipolata sausage',
    'vegetables for chipolata sausage',
    'pepper',
    'rucola',
    'oven vegetables',
    'zuchetti',
    'eggplant',
    'piccata mass',
    'vegetables for piccata',
    'chocolate cake',
    'spicy vegetable ragout',
    'tomato',
    'turkey breast',
    'lenses brown',
    'lenses',
    'pear',
    'soggy bread without crust',
    'big square plate',
    'soup of the day mushroom cream',
    'oversoaked roast beef',
    'oversoaked food in pyramid shape',
    'oversoaked chia pudding',
    'oversoaked mixed roast beef',
    'cheese sauce',
    ' swiss chard',
    'oversoaked mixed chickpea curry',
    'smoked sausage (landjaeger)',
    'soup of the day leek cream',
    'vegetables for fregola',
    'radish salad',
    'fresh cheese praline',
    'pickled cucumber',
    'soup of the day banana-coconut',
    'pickled vegetables',
    'deli meat cheese',
    'turkey',
    'cylindrical transparent shot-glass',
    'ricotta tortellini',
    'potato vegetable curry',
    'soup of the day sweetcorn',
    'baked chickpea',
    'milk coffee',
    'cherry jam',
    'coffee cup',
    'coffee plate',
    'appenzeller cheese',
    'paprika sauce',
    'bramata',
    'green spelt dumplings',
    'vegetable ragout for green spelt dumplings',
    'chicken thigh steak',
    'soup of the day tomatoes',
    'vegetable salad for ham',
    'country smoked ham',
    'lye rolls',
    'antipasti vegetables',
    'tagliatelle tomato pesto antipasti',
    'soup of the day beetroot',
    'soy yogurt dip',
    'bell pepper stew',
    'pineapple-quark-mousse',
    'quinoa salad',
    'dried tomatoes',
    'endives orange salad',
    'orange fillet',
    'mascarpone',
    'shiitake',
    'little glass bowl',
    'vegetable salad for quinoa',
    'halibut',
    'asian dip',
    'potato hash brown (roesti) with vegetables',
    'oversoaked salmon fillet',
    'oversoaked chickpea puree',
    'port wine pears rucola risotto',
    'soup of the day barley',
    'gorgonzola',
    'creamy polenta medium',
    'beef patties in juicy sauce',
    'merlot sauce',
    'oversoaked perch fillet',
    'oversoft food in crescent shape',
    'rocket risotto',
    'oversoaked bell peppers',
    'oversoakeboiled beef',
    'oversoakeboiled polenta',
    'oversoaked carrots',
    'soft zuchetti',
    'veggie cervalat sausage',
    'grisons barley soup',
    'soup of the day cauliflower cream',
    'herbal semolina slice',
    'crispy vegetable roll',
    'soup of the day parmesan foam',
    'gnocchi pan tofu',
    'homemade fishburgers',
    'soup of the day zucchetti',
    'egg vinaigrette',
    'cheesy soft egg noodles (kaesespaetzle)',
    'fennel salad for bowl',
    'tree nut dressing',
    'spelt marinated',
    'feta marinated',
    'beetroot cooked',
    'oversoaked smoked salmon',
    'softened panna cotta',
    'protein bowl',
    'oversoaked fennel',
    'oversoaked couscous',
    'oversoaked turkey plate',
    'wild rice raw',
    'homemade veggie burger',
    'champignon organic',
    'pork steak',
    'swiss macaroni and cheese',
    'cantadou cheese',
    'sliced beef',
    'knot rolls',
    'french dressing',
    'minced poultry patties',
    'carbonara',
)
@DATASETS.register_module()
class CocoDataset(BaseDetDataset):
    """Dataset for COCO."""
    
    # custom
    do_closed_set_training = False
    print('\nclosed_set_training\n' if do_closed_set_training else '\nopen set training\n')

    if do_closed_set_training:
        METAINFO = {
            'classes': ALL_LABELS
        }
    else:  # provide, extra wrong classes for open set training
        METAINFO = {
            'classes': ALL_LABELS
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
                raw_img_info,
                'closed_set_training':
                self.do_closed_set_training,
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
