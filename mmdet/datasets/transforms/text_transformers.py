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
    "apple mousse",
    "apple sauce",
    "apple tart",
    "applesauce",
    "apricot quark",
    "apricot tart",
    "apricot yogurt",
    "baby lettuce",
    "baby spinach",
    "baked meatloaf",
    "balsamic sauce",
    "bami goreng",
    "banana",
    "basil sauce",
    "beans",
    "beef braised slice",
    "beef lasagna",
    "beef meatballs",
    "beef roast",
    "beef tartare",
    "beetroot",
    "bell pepper",
    "bell pepper sauce",
    "berry",
    "bircher muesli",
    "black bean puree",
    "black olives",
    "blackened",
    "boiled beef",
    "boiled egg",
    "boiled eggs",
    "boiled potatoes",
    "bok choy",
    "bramata slice",
    "bread",
    "bread roll",
    "bread without crust",
    "breaded poultry meatball",
    "broccoli",
    "broth",
    "brownie",
    "bulgur",
    "butter",
    "cabbage salad",
    "caper butter sauce",
    "capers",
    "capuns with cheese",
    "caramel flan",
    "carbonara tofu",
    "carrot",
    "carrot appetizer",
    "carrots",
    "cashew nuts",
    "cauliflower",
    "celery salad",
    "chanterelle parsley risotto",
    "cheese",
    "cheese ball",
    "cheese crepe",
    "cheese ravioli",
    "cheese spaetzle",
    "cheese spread",
    "cheese tart",
    "cheesecake",
    "cherry tomato",
    "chicken breast",
    "chicken cordon bleu",
    "chicken thigh",
    "chickpea puree",
    "chili pepper",
    "chili with vegetables",
    "chipolata",
    "chive oil",
    "chives",
    "chocolate ice cream",
    "chocolate mousse",
    "chocolate powder bag",
    "chocolate yogourt",
    "chocolate yogurt",
    "chopped herbs",
    "cocktail sauce",
    "cod",
    "coffee",
    "coffee cream",
    "coffee yogurt",
    "cognac sauce",
    "cold cuts",
    "colorful vegetable pan with soft egg noodles (spaetzle)",
    "colorful vegetables for veggie cervalat sausage",
    "cooked beetroot",
    "corn",
    "cottage cheese",
    "cranberries",
    "cream",
    "cream herb sauce",
    "cream sauce",
    "cream slice",
    "creamed spinach",
    "cress",
    "croissant",
    "cucumber",
    "diced tomatoes",
    "dried apricots",
    "dried meat",
    "dried tomato",
    "duchess potatoes",
    "egg",
    "eggplant cordon bleu",
    "eggplant moussaka",
    "eggplant piccata",
    "emmental cheese",
    "emmentaler",
    "endive orange salad",
    "energie supplement",
    "extra protein",
    "fish burger",
    "frech salad dressing",
    "fregola",
    "french dressing",
    "french salad dressing",
    "fried onions",
    "fried rice",
    "fruit jelly",
    "fruit quark",
    "fruit salad",
    "glazed carrots",
    "gnocchi",
    "gnocchi seitan pan",
    "golden berry",
    "goulash soup",
    "grana padano",
    "grape",
    "grated cheese",
    "gravy",
    "green beans",
    "green peas",
    "gruyere cheese",
    "halloumi",
    "ham sandwich",
    "hash brown (roesti) with cheese",
    "hawaiian toast",
    "herb cream sauce",
    "herb quark dip",
    "herb rice",
    "hollandaise sauce",
    "honey",
    "horseradish foam",
    "iceberg lettuce",
    "italian sauce",
    "jam",
    "jam sandwich cookie",
    "juniper jus",
    "kiwi",
    "knot bread rolls",
    "lactose-free milk",
    "lamb stew",
    "lamb's lettuce (nuesslisalat / nuessli)",
    "leek",
    "legume salad",
    "lemon",
    "lemon slice",
    "lemon sorbet",
    "lettuce",
    "lingonberry compote",
    "lollo rosso",
    "mac and cheese",
    "macaroni",
    "margarine",
    "marinated beans",
    "mashed black beans",
    "mashed pasta",
    "mashed potato",
    "mashed potato with dill",
    "mashed potatoes",
    "mayonnaise",
    "meatloaf",
    "merlot jus",
    "milk",
    "minced beef sauce",
    "minced meat",
    "mint",
    "mixed carrot and peas",
    "mixed salad",
    "mixed vegetables",
    "mozzarella salad",
    "multigrain roll",
    "mushroom risotto",
    "mushrooms",
    "mustard",
    "mustard sauce",
    "nut cake",
    "onion sauce",
    "orange",
    "orange segment",
    "oriental rice",
    "oversoaked bean",
    "oversoaked beef roast",
    "oversoaked bell pepper",
    "oversoaked boiled potatoes",
    "oversoaked bolognese",
    "oversoaked broccoli",
    "oversoaked carrot",
    "oversoaked celery",
    "oversoaked chicken strips with sauce",
    "oversoaked chickpea curry",
    "oversoaked creamed spinach",
    "oversoaked millet slice",
    "oversoaked poached salmon",
    "oversoaked polenta",
    "oversoaked salmon",
    "oversoaked sliced chicken",
    "oversoaked zucchetti",
    "panna cotta",
    "parsley",
    "parsley fritters",
    "peanuts",
    "pear",
    "peas",
    "peeled carrot",
    "penne",
    "penne pasta",
    "peperonata",
    "pepper",
    "pernod sauce",
    "pesto cream sauce",
    "pickled cucumber",
    "pickles",
    "pizokel vegetable gratin",
    "plain yogurt",
    "plum",
    "plum crumble",
    "plum muffin",
    "plum tart",
    "plums",
    "polenta",
    "pomegranate",
    "pork steak",
    "potato",
    "potato herb patties",
    "potato wedges",
    "protein drink",
    "protein supplement",
    "pumpkin seeds",
    "pureed beef",
    "pureed bratwurst",
    "pureed carrot",
    "pureed omelette",
    "quail breast",
    "quince jelly",
    "quinoa",
    "quinoa patty",
    "radicchio rosso",
    "radish",
    "rasberry",
    "raspberry",
    "raspberry quark",
    "raspberry yogurt lactose-free",
    "raw egg",
    "red cabbage",
    "red chicory",
    "red onion",
    "red wine",
    "rice",
    "rice noodle salad",
    "risotto",
    "romanesco",
    "root vegetables",
    "rosemary sauce",
    "rusk",
    "rustico croissant",
    "sachertorte",
    "saffron risotto",
    "salad",
    "salad dressing",
    "salmon",
    "salmon cube",
    "salt",
    "sardinian fregola",
    "sauce",
    "sausage cheese salad",
    "sauteed tomato",
    "scrambled eggs",
    "sea bass",
    "shrimps",
    "sliced quorn",
    "sliced turkey breast",
    "sliced veal",
    "smoked sausage",
    "smoked trout",
    "snow peas",
    "soft cheese",
    "soft egg noodles (spaetzle)",
    "sole",
    "soup",
    "sour cream",
    "special bean",
    "spelt dumplings",
    "spelt goulash",
    "spelt risotto",
    "spicy tomato vegetable sauce",
    "spinach",
    "spinach leaf",
    "spinach tart",
    "spreadable cheese",
    "spring onion",
    "spring onions",
    "stawberry yogourt",
    "strawberry ice cream",
    "strawberry yogurt",
    "sugar",
    "swedish cake",
    "sweet potato",
    "swiss chard vegetable ragout",
    "tagliatelle",
    "thai glass noodle salad",
    "thin chocolate",
    "thin chocolate decoration",
    "tilapia",
    "tilsiter",
    "tiramisu",
    "tiramisu slice",
    "toast bread",
    "toasted bread",
    "tofu",
    "tomato",
    "tomato cream sauce",
    "tomato sauce",
    "tortellini",
    "trout fillet",
    "trout tartare",
    "turnip cabbage",
    "vanilla cream",
    "vanilla ice cream",
    "vanilla porridge",
    "veal cheek",
    "veal sausage",
    "veal steak",
    "vegan meatballs",
    "vegetable",
    "vegetable bolognese",
    "vegetable lasagna",
    "vegetable piccata",
    "vegetable ragout",
    "vegetable salad",
    "vegetable stew",
    "vegetable strudel",
    "vegetables",
    "vegetables for boiled meat salad",
    "vegetables for meatballs",
    "vegetarian burger",
    "veggie crispy bites",
    "veggie sausage",
    "walnut",
    "walnut dressing",
    "walnut pesto sauce",
    "whipped cream",
    "white bean puree",
    "white wine",
    "white wine sauce",
    "wild rice",
    "wine",
    "yeast roll",
    "yellow pea puree",
    "zucchini",
    "\"salade nicoise\"",
    "apple juice",
    "aufschnittteller",
    "aufschnittteller vvg",
    "bacon",
    "bag of ovaltine",
    "bean cassoulet",
    "beet ginger salad",
    "boiled beef salad",
    "bread dumpling",
    "brie cheese",
    "broth for halibut",
    "burrito",
    "buttered pretzel",
    "cabbage",
    "cacao powder",
    "capuns",
    "carrot puree",
    "celery",
    "cheese sandwich",
    "cherry compote",
    "chervil cream sauce",
    "chicken",
    "chicken breast slices",
    "chickpea triangles",
    "chocolate drink",
    "choernlibroetli",
    "cinnamon sugar",
    "cod with herbs",
    "colorful spaetzle",
    "country cuts",
    "cranberry",
    "cream tart",
    "crispy fried onions",
    "croutons",
    "curry sauce",
    "endive",
    "energy cream",
    "finger-shaped potato dumplings",
    "fruit",
    "green salad",
    "gruyere",
    "halibut",
    "herb semolina slice",
    "horseradish bouillon",
    "hummus",
    "ketchup",
    "kohlrabi",
    "lactose free dessert",
    "lamb loin",
    "lemon roulade",
    "lentil ragout",
    "mashed peas",
    "mashed semolina",
    "muffin",
    "mushroom cream sauce",
    "nuts",
    "olive",
    "orange juice",
    "oversoaked cauliflower",
    "oversoaked cheese plate",
    "oversoaked scrambled egg",
    "pasta",
    "pasta with tomato sauce",
    "peppermint",
    "pickle",
    "pilaf rice",
    "pita bread",
    "poultry ragout",
    "pureed chicken",
    "pureed polenta",
    "quark",
    "quorn strips in cream sauce",
    "ratatouille",
    "red pepperoncini",
    "roast beef",
    "roasted cashew nuts",
    "roll bread",
    "root vegetable",
    "rye bread",
    "roesti",
    "salad leaves",
    "salami",
    "salami sandwich",
    "salt and pepper",
    "salted potatoes",
    "sauerkraut",
    "scrambled egg",
    "seitan strips",
    "semolina porridge",
    "sesame tofu",
    "smoked pork neck",
    "snow peas with carrots",
    "soft bircher muesli",
    "spanish tortilla",
    "special jus",
    "springroll",
    "sprout vegetable",
    "strawberry quark",
    "sugar peas",
    "sweet and sour carrot",
    "tea",
    "turkey cold cut",
    "vanilla ice",
    "vegan nuggets",
    "vegetable curry",
    "vegetable salad with feta",
    "vegetable salad with white beans",
    "vegetables for halibut",
    "whole grain rice cake",
    "apricot yoghurt",
    "arugula",
    "balsamic dressing",
    "beef",
    "birchermuesli",
    "bolognaise",
    "bolognese",
    "brie",
    "chocolate",
    "chocolate bar",
    "compote",
    "cured ham",
    "dip",
    "dressing",
    "eggplant",
    "fish",
    "hash brown (roesti)",
    "herb potato patty",
    "italian dressing",
    "lard",
    "lasagna",
    "lye bread",
    "milk coffee",
    "other food",
    "paprika sauce",
    "protein powder",
    "pureed chickpeas",
    "ravioli",
    "ruccola",
    "sandwich",
    "sausage",
    "sliced quorn sauce",
    "spaghetti",
    "tart",
    "toast",
    "tomato vegetable sauce",
    "turkey breast",
    "wedges",
    "yogourt plain",
    "yogurt",
    "almost empty",
    "bead",
    "bellpeper",
    "bramata",
    "brocoli",
    "chocolate icecream",
    "cream cheese",
    "fresh cheese praline",
    "herb cream",
    "herbs cheese bite",
    "macaroni and cheese",
    "milkcoffee",
    "mustard greens",
    "ovomaltine",
    "porridge",
    "pureed cauliflower",
    "pureed salmon",
    "radish salad",
    "raspberry yogurt",
    "smoked salmon",
    "blueberry",
    "cold cut meatloaf",
    "fruit yoghurt",
    "lollo green",
    "potato dumplings",
    "sour cabbage",
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
