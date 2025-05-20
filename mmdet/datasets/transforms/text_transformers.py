# Copyright (c) OpenMMLab. All rights reserved.
import json

from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes

try:
    from transformers import AutoTokenizer
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

import random
import re

import numpy as np


ALL_LABELS = (
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


def clean_name(name):
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    name = name.lower()
    return name


def check_for_positive_overflow(gt_bboxes, gt_labels, text, tokenizer,
                                max_tokens):
    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    positive_label_list = np.unique(gt_labels).tolist()
    # random shuffule so we can sample different annotations
    # at different epochs
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):

        label_text = clean_name(text[str(label)]) + '. '

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_tokens:
            break
        else:
            kept_lables.append(label)

    keep_box_index = []
    keep_gt_labels = []
    for i in range(len(gt_labels)):
        if gt_labels[i] in kept_lables:
            keep_box_index.append(i)
            keep_gt_labels.append(gt_labels[i])

    return gt_bboxes[keep_box_index], np.array(
        keep_gt_labels, dtype=np.long), length


def generate_senetence_given_labels(positive_label_list, negative_label_list,
                                    text):
    label_to_positions = {}

    label_list = negative_label_list + positive_label_list

    random.shuffle(label_list)

    pheso_caption = ''

    label_remap_dict = {}
    for index, label in enumerate(label_list):

        start_index = len(pheso_caption)

        pheso_caption += clean_name(text[str(label)])

        end_index = len(pheso_caption)

        if label in positive_label_list:
            label_to_positions[index] = [[start_index, end_index]]
            label_remap_dict[int(label)] = index

        # if index != len(label_list) - 1:
        #     pheso_caption += '. '
        pheso_caption += '. '

    return label_to_positions, pheso_caption, label_remap_dict


@TRANSFORMS.register_module()
class RandomSamplingNegPos(BaseTransform):

    def __init__(self,
                 tokenizer_name,
                 num_sample_negative=85,
                 max_tokens=256,
                 full_sampling_prob=0.5,
                 label_map_file=None):
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_sample_negative = num_sample_negative
        self.full_sampling_prob = full_sampling_prob
        self.max_tokens = max_tokens
        self.label_map = None
        if label_map_file:
            with open(label_map_file, 'r') as file:
                self.label_map = json.load(file)

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            return self.vg_aug(results)
        else:
            return self.od_aug(results)

    def vg_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text'].lower().strip()
        if not text.endswith('.'):
            text = text + '. '

        phrases = results['phrases']
        # TODO: add neg
        positive_label_list = np.unique(gt_labels).tolist()
        label_to_positions = {}
        for label in positive_label_list:
            label_to_positions[label] = phrases[label]['tokens_positive']

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = text
        results['tokens_positive'] = label_to_positions
        return results

    def od_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']

        if 'text' not in results:
            assert self.label_map is not None
            text = self.label_map
        else:
            text = results['text']

        original_box_num = len(gt_labels)
        # If the category name is in the format of 'a/b' (in object365),
        # we randomly select one of them.
        for key, value in text.items():
            if '/' in value:
                text[key] = random.choice(value.split('/')).strip()

        gt_bboxes, gt_labels, positive_caption_length = \
            check_for_positive_overflow(gt_bboxes, gt_labels,
                                        text, self.tokenizer, self.max_tokens)

        if len(gt_bboxes) < original_box_num:
            print('WARNING: removed {} boxes due to positive caption overflow'.
                  format(original_box_num - len(gt_bboxes)))

        valid_negative_indexes = list(text.keys())

        positive_label_list = np.unique(gt_labels).tolist()
        full_negative = self.num_sample_negative

        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        outer_prob = random.random()

        if outer_prob < self.full_sampling_prob:
            # c. probability_full: add both all positive and all negatives
            num_negatives = full_negative
        else:
            if random.random() < 1.0:
                num_negatives = np.random.choice(max(1, full_negative)) + 1
            else:
                num_negatives = full_negative

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)

            for i in np.random.choice(
                    valid_negative_indexes, size=num_negatives, replace=False):
                if int(i) not in positive_label_list:
                    negative_label_list.add(i)

        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)
        random.shuffle(negative_label_list)

        negative_max_length = self.max_tokens - positive_caption_length
        screened_negative_label_list = []

        for negative_label in negative_label_list:
            label_text = clean_name(text[str(negative_label)]) + '. '

            tokenized = self.tokenizer.tokenize(label_text)

            negative_max_length -= len(tokenized)

            if negative_max_length > 0:
                screened_negative_label_list.append(negative_label)
            else:
                break
        negative_label_list = screened_negative_label_list
        label_to_positions, pheso_caption, label_remap_dict = \
            generate_senetence_given_labels(positive_label_list,
                                            negative_label_list, text)

        # label remap
        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions

        return results


@TRANSFORMS.register_module()
class LoadTextAnnotations(BaseTransform):
    
    #ToDo: refractor this code, replase method choose_n_based_on_probabilities,
    #define probabilities as input (and normalize them to be sure that they sum up to one)
    def choose_n_based_on_probabilities(self, probabilities):
        random_value = random.uniform(0, 1)
        cumulative_probability = 0.0
        for n, probability in probabilities.items():
            cumulative_probability += probability
            if random_value <= cumulative_probability:
                return n
        return 0  # Fallback in case of rounding issues
    
    def get_extra_classes(self, true_classes: tuple, all_classes: tuple) -> tuple:
        # Define probabilities for choosing n wrong labels with uniform distribution
        # Allow up to 15 additional labels
        max_extra_labels = 15
        probabilities = {i: 1/(max_extra_labels+1) for i in range(max_extra_labels+1)}
        
        # Choose a number n based on the defined probabilities
        n = self.choose_n_based_on_probabilities(probabilities)
        
        # Choose n random wrong labels from the set of all labels
        filtered_classes = [c for c in all_classes if c not in true_classes]
        extra_classes = random.sample(filtered_classes, min(n, len(filtered_classes)))
        
        return extra_classes
    
    #def get_tokens_positive_for_aug_text(aug_text_data: tuple) -> list:
        #aug_tokens_positive = []
        
        # each class name is treated as a separate sentence
        #for class_name in aug_text_data:  
            #aug_tokens_positive.append([[0, len(class_name)]])
        #return aug_tokens_positive

    def transform(self, results: dict) -> dict:
            
        # same classes as in dataset.metadata.classes 
        all_classes_augmented = ALL_LABELS
        
        if 'phrases' in results:
            tokens_positive = [
                phrase['tokens_positive']
                for phrase in results['phrases'].values()
            ]
            results['tokens_positive'] = tokens_positive
        else:                      
            # Extract true classes from annotations (original text data)
            true_classes = results['text'] 
            #print('raw text data:', true_classes)
            #print('raw text data type:', type(true_classes))
            
            # Augment the current text data
            extra_classes = self.get_extra_classes(true_classes, all_classes_augmented)
            
            # Combine true classes and extra classes into a single tuple (augmented text data)
            results['text'] = true_classes + tuple(extra_classes)
            #print('augmented text data:', results['text'])
            #print('augmented text data type:', type(results['text']))
            
            # Create list of tokens_positive for all the classes in the text prompt
            #results['tokens_positive'] = get_tokens_positive_for_aug_text(results['text'])
        
        return results
