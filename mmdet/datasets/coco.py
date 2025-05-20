# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset

#ToDo: remove labels here, they can be set dynamically in the config


ALL_LABELS =(
   "apple",
    "apple cookie",
    "applesauce",
    "apricot yoghurt",
    "apricot yogurt",
    "arugula",
    "balsamic dressing",
    "bami goreng",
    "beans",
    "beef",
    "beet ginger salad",
    "bell pepper",
    "bircher muesli",
    "birchermuesli",
    "boiled beef",
    "boiled beef salad",
    "bolognaise",
    "bolognese",
    "bramata slice",
    "bread",
    "bread dumpling",
    "bread roll",
    "breaded poultry meatball",
    "brie",
    "broccoli",
    "broth",
    "brownie",
    "bulgur",
    "burrito",
    "butter",
    "cabbage salad",
    "capers",
    "caramel flan",
    "carbonara tofu",
    "carrot",
    "cashew nuts",
    "cauliflower",
    "cheese",
    "cheese crepe",
    "cheese ravioli",
    "cheesecake",
    "cherry tomato",
    "chicken",
    "chickpea puree",
    "chickpea triangles",
    "chipolata",
    "chocolate",
    "chocolate bar",
    "chocolate drink",
    "chocolate ice cream",
    "chocolate mousse",
    "chocolate yogurt",
    "cocktail sauce",
    "cod",
    "coffee",
    "coffee cream",
    "compote",
    "cream",
    "cream sauce",
    "cream slice",
    "croissant",
    "cucumber",
    "cured ham",
    "curry sauce",
    "diced tomatoes",
    "dip",
    "dressing",
    "egg",
    "eggplant",
    "eggplant moussaka",
    "fish",
    "french dressing",
    "french salad dressing",
    "fruit quark",
    "fruit salad",
    "goulash soup",
    "grana padano",
    "grated cheese",
    "gravy",
    "green beans",
    "gruyere",
    "hash brown (roesti)",
    "hawaiian toast",
    "herb potato patty",
    "horseradish foam",
    "hummus",
    "italian dressing",
    "jam",
    "lard",
    "lasagna",
    "leek",
    "legume salad",
    "lemon",
    "lemon sorbet",
    "lettuce",
    "lollo rosso",
    "lye bread",
    "mashed peas",
    "mashed potato",
    "mashed potatoes",
    "meatloaf",
    "milk",
    "milk coffee",
    "minced beef sauce",
    "mint",
    "mixed salad",
    "multigrain roll",
    "mushrooms",
    "mustard",
    "nut cake",
    "olive",
    "orange",
    "orange juice",
    "other food",
    "panna cotta",
    "paprika sauce",
    "parsley",
    "parsley fritters",
    "pasta",
    "peas",
    "pepper",
    "pizokel vegetable gratin",
    "plain yogurt",
    "plum crumble",
    "plum muffin",
    "polenta",
    "pork steak",
    "potato",
    "poultry ragout",
    "protein powder",
    "pureed bratwurst",
    "pureed carrot",
    "pureed chickpeas",
    "pureed omelette",
    "quinoa",
    "radicchio rosso",
    "radish",
    "rasberry",
    "raspberry",
    "ratatouille",
    "ravioli",
    "rice",
    "rice noodle salad",
    "risotto",
    "romanesco",
    "ruccola",
    "salad",
    "salad dressing",
    "salami",
    "salmon",
    "sandwich",
    "sauce",
    "sausage",
    "sausage cheese salad",
    "scrambled eggs",
    "sliced quorn sauce",
    "sliced veal",
    "soft cheese",
    "soft egg noodles (spaetzle)",
    "soup",
    "sour cream",
    "spaghetti",
    "spinach",
    "spring onions",
    "strawberry ice cream",
    "sugar peas",
    "sweet potato",
    "swiss chard vegetable ragout",
    "tart",
    "tea",
    "thai glass noodle salad",
    "tiramisu",
    "toast",
    "tomato",
    "tomato sauce",
    "tomato vegetable sauce",
    "tortellini",
    "turkey breast",
    "vanilla cream",
    "vanilla ice cream",
    "vegan meatballs",
    "vegetable bolognese",
    "vegetable curry",
    "vegetable piccata",
    "vegetable ragout",
    "vegetable salad",
    "vegetables",
    "vegetarian burger",
    "wedges",
    "whipped cream",
    "yogourt plain",
    "yogurt",
    "zucchini",
    "almost empty",
    "bacon",
    "banana",
    "bead",
    "bean cassoulet",
    "beef braised slice",
    "beef meatballs",
    "beef roast",
    "bellpeper",
    "black bean puree",
    "bok choy",
    "bramata",
    "bread without crust",
    "brocoli",
    "capuns",
    "chili with vegetables",
    "chives",
    "chocolate icecream",
    "cinnamon sugar",
    "corn",
    "cream cheese",
    "croutons",
    "endive orange salad",
    "fregola",
    "fresh cheese praline",
    "fried onions",
    "herb cream",
    "herb semolina slice",
    "herbs cheese bite",
    "honey",
    "kohlrabi",
    "lamb stew",
    "lemon roulade",
    "macaroni and cheese",
    "margarine",
    "mashed pasta",
    "milkcoffee",
    "muffin",
    "mustard greens",
    "nuts",
    "ovomaltine",
    "peperonata",
    "pickle",
    "pickled cucumber",
    "pita bread",
    "pomegranate",
    "porridge",
    "protein drink",
    "pureed beef",
    "pureed cauliflower",
    "pureed chicken",
    "pureed salmon",
    "quinoa patty",
    "radish salad",
    "raspberry yogurt",
    "red chicory",
    "roll bread",
    "rye bread",
    "sardinian fregola",
    "seitan strips",
    "smoked salmon",
    "spanish tortilla",
    "spelt dumplings",
    "spelt goulash",
    "spring onion",
    "springroll",
    "strawberry yogurt",
    "tilsiter",
    "tofu",
    "turkey cold cut",
    "veal cheek",
    "vegetable strudel",
    "veggie crispy bites",
    "white bean puree",
    "yeast roll",
    "apple juice",
    "bag of ovaltine",
    "beef tartare",
    "beetroot",
    "buttered pretzel",
    "caper butter sauce",
    "carrot appetizer",
    "celery",
    "cheese plate",
    "chicken cordon bleu",
    "chili pepper",
    "chocolate powder bag",
    "coffee yogurt",
    "cold cuts",
    "cottage cheese",
    "country cuts",
    "cranberry",
    "cress",
    "dried tomato",
    "duchess potatoes",
    "eggplant cordon bleu",
    "eggplant piccata",
    "emmental cheese",
    "fish burger",
    "fruit",
    "gnocchi",
    "gnocchi seitan pan",
    "golden berry",
    "grape",
    "gruyere",
    "halloumi",
    "herbs",
    "hollandaise sauce",
    "horseradish bouillon",
    "jam sandwich cookie",
    "ketchup",
    "kiwi",
    "lamb",
    "lentil ragout",
    "mashed black beans",
    "mashed semolina",
    "mayonnaise",
    "millet slice",
    "oil",
    "olives",
    "onion",
    "peanuts",
    "pear",
    "peeled carrot",
    "plum",
    "potato dumplings",
    "pureed polenta",
    "quail breast",
    "quark",
    "red cabbage",
    "red pepperoncini",
    "rusk",
    "sachertorte",
    "shrimps",
    "sliced quorn",
    "smoked trout",
    "sour cabbage",
    "swedish cake",
    "sweet and sour carrot",
    "thin chocolate decoration",
    "trout tartare",
    "veal steak",
    "vegetable stew",
    "walnut",
    "whole grain rice cake",
    "wine",
    "apple mousse",
    "apple sauce",
    "apricot quark",
    "balsamic sauce",
    "berry",
    "blackened",
    "chanterelle parsley risotto",
    "cheese ball",
    "chocolate yogourt",
    "colorful vegetables for veggie cervalat sausage",
    "dried apricots",
    "dried meat",
    "emmentaler",
    "energie supplement",
    "extra protein",
    "frech salad dressing",
    "green peas",
    "italian sauce",
    "lamb's lettuce (nuesslisalat / nuessli)",
    "mozzarella salad",
    "pickles",
    "plums",
    "protein supplement",
    "pumpkin seeds",
    "raw egg",
    "salt",
    "snow peas",
    "stawberry yogourt",
    "sugar",
    "toasted bread",
    "turnip cabbage",
    "\"salade nicoise\"",
    "aufschnittteller",
    "aufschnittteller vvg",
    "broth for halibut",
    "cabbage",
    "cacao powder",
    "choernlibroetli",
    "endive",
    "energy cream",
    "lactose free dessert",
    "oversoaked cauliflower",
    "peppermint",
    "quorn strips in cream sauce",
    "roesti",
    "salt and pepper",
    "scrambled egg",
    "vanilla ice",
    "vegan nuggets",
    "baked chickpea",
    "chocolate cake",
    "mozzarella",
    "thyme",
    "tilster cheese",
    "turmeric",
    "small sauce-glass",
    "white plate",
    "shallow bowl",
    "caesar salad",
    "parmesan dressing",
    "egg cooked",
    "toast croutons",
    "parmesan shavings",
    "white plate without rim",
    "soup of the day ratatouille cream",
    "soup-bowl",
    "finger-shaped potato dumplings (schupfnudeln)",
    "hioumi",
    "natural yogurt",
    "beans green",
    "horn-shaped pasta (hoernli)",
    "big sauce-glass",
    "soup of the day artichoke",
    "glass fruitsalad-bowl",
    "vanilla cream puffs",
    "small quadratic plate-bowl",
    "basil pesto",
    "quadratic dessert-plate",
    "sausage and cheese salad",
    "lollo bianco",
    "house bread",
    "soup of the day potato",
    "vegetarian bami goreng",
    "soup of the day broccoli cream",
    "vegetable strips",
    "saffron herb sauce",
    "vegetable strips saffron-herb sauce",
    "pureed green balls",
    "pureed food in a special shape",
    "pureed meat slices",
    "pureed food in oval shape",
    "pureed broccoli",
    "pureed mashed potatoes",
    "sprout vegetables",
    "scallion",
    "herbal rice",
    "soup of the day curry cream",
    "paneer",
    "quinoa patties",
    "vegetables for quinoa patties",
    "turkey ham",
    "pineapple",
    "onion red",
    "barley risotto",
    "lemon panna cotta",
    "soup of the day yellow pea",
    "dill mashed potatoes",
    "salmon cubes marinated",
    "brown sauce",
    "pureed food in pyramid shape",
    "penne rigate",
    "pureed balls",
    "raspberry mousse in pyramid shape",
    "white sauce",
    "round raspberry mousse",
    "slices",
    "poultry stew",
    "pureed chicken thigh",
    "pureed fries",
    "pureed sausage",
    "plate with red rim",
    "veggie swiss macaroni and cheese",
    "poulet",
    "boiled meat salad seed oil",
    "vegetable patch",
    "boiled meat",
    "currant sheet cake",
    "bulgur sauce",
    "sliced seitan",
    "oyster mushrooms",
    "vegetables for green spelt risotto",
    "green spelt risotto",
    "bouillon",
    "cold chicken breast",
    "soup of the day carrot cream",
    "curry dip",
    "soup of the day lentil ginger",
    "poulet cordon bleu",
    "pilau rice",
    "roasted cauliflower",
    "sauce for sliced seitan",
    "small plastic cup",
    "overly soft thick brie cheese",
    "overly soft cottage cheese",
    "meat cheese",
    "lyonnaise potatoes",
    "oversoaked sliced veal",
    "overly soft cream cheese",
    "overly soft thin brie cheese",
    "currants",
    "soup of the day bell peppers",
    "sliced quorn zurich style",
    "colorful vegetables from zuchetti peas carrots and beans",
    "bread dumplings",
    "sauce poultry ragout",
    "banana organic",
    "lye croissant",
    "lid on the ground",
    "uncovered jug",
    "jug covered with lid",
    "mueesli",
    "large glass fruitsalad-bowl",
    "milk roll",
    "baked vegetables for mozzarella",
    "chipolata sausage",
    "rucola",
    "oven vegetables",
    "zuchetti",
    "piccata mass",
    "vegetables for piccata",
    "spicy vegetable ragout",
    "lenses brown",
    "lenses",
    "soggy bread without crust",
    "big square plate",
    "soup of the day mushroom cream",
    "oversoaked roast beef",
    "oversoaked food in pyramid shape",
    "oversoaked chia pudding",
    "oversoaked mixed roast beef",
    "cheese sauce",
    " swiss chard",
    "oversoaked mixed chickpea curry",
    "smoked sausage (landjaeger)",
    "soup of the day leek cream",
    "vegetables for fregola",
    "soup of the day banana-coconut",
    "pickled vegetables",
    "deli meat cheese",
    "turkey",
    "cylindrical transparent shot-glass",
    "ricotta tortellini",
    "potato vegetable curry",
    "soup of the day sweetcorn",
    "cherry jam",
    "coffee cup",
    "coffee plate",
    "appenzeller cheese",
    "green spelt dumplings",
    "vegetable ragout for green spelt dumplings",
    "chicken thigh steak",
    "soup of the day tomatoes",
    "vegetable salad for ham",
    "country smoked ham",
    "lye rolls",
    "antipasti vegetables",
    "tagliatelle tomato pesto antipasti",
    "soup of the day beetroot",
    "bell pepper stew",
    "pineapple-quark-mousse",
    "quinoa salad",
    "dried tomatoes",
    "endives orange salad",
    "orange fillet",
    "mascarpone",
    "shiitake",
    "little glass bowl",
    "vegetable salad for quinoa",
    "asian dip",
    "potato hash brown (roesti) with vegetables",
    "oversoaked salmon fillet",
    "oversoaked chickpea puree",
    "port wine pears rucola risotto",
    "soup of the day barley",
    "gorgonzola",
    "creamy polenta medium",
    "beef patties in juicy sauce",
    "merlot sauce",
    "oversoaked perch fillet",
    "oversoft food in crescent shape",
    "rocket risotto",
    "oversoaked bell peppers",
    "oversoakeboiled beef",
    "oversoakeboiled polenta",
    "oversoaked carrots",
    "soft zuchetti",
    "veggie cervalat sausage",
    "grisons barley soup",
    "soup of the day cauliflower cream",
    "herbal semolina slice",
    "crispy vegetable roll",
    "soup of the day parmesan foam",
    "gnocchi pan tofu",
    "homemade fishburgers",
    "soup of the day zucchetti",
    "egg vinaigrette",
    "cheesy soft egg noodles (kaesespaetzle)",
    "fennel salad for bowl",
    "tree nut dressing",
    "spelt marinated",
    "feta marinated",
    "beetroot cooked",
    "oversoaked smoked salmon",
    "softened panna cotta",
    "protein bowl",
    "oversoaked fennel",
    "oversoaked couscous",
    "oversoaked turkey plate",
    "wild rice raw",
    "homemade veggie burger",
    "champignon organic",
    "swiss macaroni and cheese",
    "cantadou cheese",
    "sliced beef",
    "knot rolls",
    "minced poultry patties",
    "carbonara",
    "blueberry",
    "cold cut meatloaf",
    "fruit yoghurt",
    "lollo green",
    "strawberry yoghurt",
    "yoghurt",
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
