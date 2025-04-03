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


ALL_LABELS = ('Other',
    'banana', 'caramel flan', 'lamb stew', 'carrots', 'bread',
    'soup', 'chanterelle parsley risotto', 'panna cotta', 'french salad dressing', 'corn',
    'pomegranate', 'chocolate ice cream', 'applesauce', 'bulgur', 'gnocchi seitan pan',
    'whipped cream', 'Plum muffin', 'grated cheese', 'lettuce', 'mashed potatoes',
    'mashed potato', 'oversoaked beef roast', 'peas', 'carrot', 'chipolata',
    'potato wedges', 'italian sauce', 'turnip cabbage', 'vanilla ice cream', 'apple',
    'fruit salad', 'mozzarella salad', 'zucchini', 'pureed carrot', 'pureed beef',
    'cream sauce', 'lemon slice', 'caper butter sauce', 'potatoes', 'sole',
    'beans', 'cherry tomato', 'mixed salad', 'meatloaf', 'cognac sauce',
    'herb cream sauce', 'eggplant moussaka', 'strawberry yogurt', 'hollandaise sauce', 'legume salad',
    'macaroni', 'tomato sauce', 'vanilla porridge', 'apricot yogurt', 'cream slice',
    'butter', 'gravy', 'bircher muesli', 'boiled egg', 'Cherry Tomato',
    'vegetables for boiled meat salad', 'shrimps', 'walnut', 'cheese spaetzle', 'cabbage salad',
    'baby lettuce', 'radish', 'orange', 'cocktail sauce', 'parsley',
    'fried onions', 'vegetable strudel', 'herb quark dip', 'chocolate yogurt', 'mixed carrot and peas',
    'rice', 'coffee yogurt', 'pureed bratwurst', 'vanilla cream', 'herb rice',
    'trout fillet', 'spinach leaf', 'thin chocolate decoration', 'salmon cubes', 'oversoaked millet slice',
    'zucchetti', 'oversoaked broccoli', 'oversoaked sliced chicken', 'oversoaked bean', 'rustico croissant',
    'multigrain roll', 'honey', 'protein drink', 'cheese crepe', 'swiss chard vegetable ragout',
    'mustard sauce', 'mint', 'moustard sauce', 'baked meatloaf', 'lingonberry compote',
    'Quail Breast', 'Cranberries', 'Celery Salad', 'jam', 'bread roll',
    'croissant', 'milk', 'chocolate powder bag', 'sliced veal', 'romanesco',
    'oversoaked zucchetti', 'polenta', 'broccoli', 'beef braised slice', 'tiramisu slice',
    'merlot jus', 'thin chocolate', 'balsamic sauce', 'strawberry ice cream', 'juniper jus',
    'veal cheak', 'fregola', 'red cabbage', 'red chicory', 'cooked beetroot',
    'orange segments', 'walnuts', 'peeled carrot', 'iceberg lettuce', 'walnut dressing',
    'chocolate mousse', 'grapes', 'cheese', 'salad', 'carrot appetizer',
    'tilapia', 'pernod sauce', 'white bean puree', 'vegetable', 'nut cake',
    'chopped herbs', 'tiramisu', 'boiled potatoes', 'cheese spread', 'minced meat',
    'oversoaked chicken strips with sauce', 'penne pasta', 'scrambled eggs', 'creamed spinach', 'rice noodle salad',
    'chili pepper', 'peanuts', 'lactose-free milk', 'Jam sandwich cookie', 'chicken breast',
    'yellow pea puree', 'plain yogurt', 'apple tart', 'cream', 'veggie crispy bites',
    'oversoaked polenta', 'Pork Steak', 'soft egg noodles (spaetzle)', 'Root Vegetables', 'vegetables for meatballs',
    'vegan meatballs', 'Tagliatelle', 'sauce', 'vegetables', 'cheese ravioli',
    'lemon sorbet', 'cashew nuts', 'smoked sausage', 'mixed vegetables', 'sardinian fregola',
    'sliced quorn', 'basil sauce', 'beef lasagna', 'glazed carrots', 'beef roast',
    'special bean', 'saffron risotto', 'bell pepper sauce', 'eggplant cordon bleu', 'peperonata',
    'oversoaked creamed spinach', 'white wine', 'cod', 'spinach tart', 'breaded poultry meatball',
    'rosemary sauce', 'cauliflower', 'yeast roll', 'oversoaked bolognese', 'chickpea puree',
    'bread without crust', 'plum', 'penne', 'diced tomatoes', 'carbonara tofu',
    'kiwi', 'pizokel vegetable gratin', 'plum tart', 'rusk', 'coffee',
    'emmental cheese', 'margarine', 'hash brown (roesti) with cheese', 'orange segment', 'oversoaked celery',
    'ham sandwich', 'sweet potato', 'soft cheese', 'spreadable cheese', 'fruit jelly',
    'quince jelly', 'cottage cheese', 'wild rice', 'spicy tomato vegetable sauce', 'vegetarian burger',
    'mushrooms', 'sea bass', 'bok choy', 'oriental rice', 'boiled beef',
    'root vegetables', 'broth', 'plum crumble', 'mayonnaise', 'greyerzer cheese',
    'apricose quark', 'veggie sausage', 'potato herb patties', 'colorful vegetables for veggie cervalat sausage', 'swedish cake',
    'golden berry', 'oversoaked boiled potatoes', 'chicken thigh', 'bramata slice', 'cream herb sauce',
    'vegetable ragout', 'spelt dumplings', 'brownie', 'fruit quark', 'gran padrone',
    'thai glass noodle salad', 'knot bread rolls', 'red wine', 'leek', 'colorful vegetable pan with soft egg noodles (spaetzle)',
    'Cashew Nuts', 'Halloumi', 'sauteed tomato', 'tofu', 'raspberry quark',
    'mac and cheese', 'gruyere cheese', 'coffee cream', 'spinach', 'sliced turkey breast',
    'tomato cream sauce', 'tortellini', 'oversoaked carrot', 'beef tartare', 'raspberry yogurt lactose-free',
    'quinoa', 'vegetable salad', 'apricot tart', 'oversoaked salmon', 'endive orange salad',
    'risotto', 'fish burger', 'black bean puree', 'mashed black beans', 'apple cookie',
    'chives', 'bread dumpling', 'poultry ragout', 'sugar peas', 'oversoaked scrambled egg',
    'potato', 'Bread', 'scrambled egg', 'sesame tofu', 'rosti',
    'quorn strips in cream sauce', 'oversoaked cauliflower', 'pureed polenta', 'apple juice', 'cream tart',
    'chervil cream sauce', 'croutons', 'bacon', 'salad dressing', 'gran padano',
    'orange juice', 'strawberry quark', 'tilster cheese', 'Chicken', 'pasta',
    'grape', 'vegetable bolognese', 'cinnamon sugar', 'cherry compote', 'semolina porridge',
    'ketchup', 'buttered pretzel', 'beet ginger salad', 'salmon', 'cholocate drink',
    'lemon roulade', 'pilaw rice', 'sour cream', 'burrito', 'country cuts',
    'green beans', 'vegan  nuggets', 'pilaf rice', 'pureed omelette', 'mashed semolina',
    'fruit', 'oversoaked bell pepper', 'oversoaked chickpea curry', 'Aufschnittteller VVG', 'bean cassoulet',
    'salad leaves', 'Choernlibroetli', 'Aufschnittteller', 'Bean Cassoulet', 'cucumber',
    'salami', 'turkey cold cut', 'boiled eggs', 'roast beef', 'vinaigrette',
    'olives', 'Halibut', 'vegetables for halibut', 'pasta with tomato sauce', 'hummus',
    'whole grain rice cake', 'tea', 'horseradish bouillon', 'salted potatoes', 'root vegetable',
    'lamb loin', 'special jus', 'mustard', 'bag of Ovaltine', 'cheese sandwich',
    'bell pepper', 'cod with herbs', 'ratatouille', 'herb semolina slice', 'carrot puree',
    'springroll', 'halloumi', 'roasted cashew nuts', 'colorful spaetzle', 'cheese tart',
    'rye bread', 'radishes', 'brie cheese', 'gruyere', 'crispy fried onions',
    'chickpea triangles', 'vegetable curry', 'red pepperoncini', 'spring onion', 'oversoaked cheese plate',
    'gnocchi', 'celery', '"Salade nicoise"', 'minced beef sauce', 'vegetable salad with white beans',
    'chicken breast slices', 'curry sauce', 'bami goreng', 'finger-shaped potato dumplings', 'smoked pork neck',
    'sauerkraut', 'mashed pasta', 'pureed chicken', 'Red Onion', 'Cranberry',
    'hawaiian toast', 'Seitan Strips', 'Sauce', 'Spanish Tortilla', 'veal sausage',
    'vegetable salad with feta', 'mashed peas', 'lentil ragout', 'pita bread', 'sprout vegetable',
    'muffin', 'soft bircher muesli', 'quark', 'sweet and sour carrot', 'salami sandwich',
    'mushroom cream sauce', 'capuns', 'Beetroot', 'Wholemeal bread', 'Jam',
    'Water', 'Banana', 'Soft cheese', 'Raw ham', 'Hard cheese',
    'Cottage cheese', 'Coffee', 'Mixed fruit', 'Pancake', 'Tea',
    'Smoked salmon', 'Avocado', 'Spring onion', 'Ristretto', 'Ham',
    'Egg', 'french fries', 'chicken', 'tomato', 'shrimp',
    'chickpeas', 'french dressing', 'horn shaped pasta', 'pear', 'cashew',
    'almonds', 'lentil', 'peanut butter', 'blueberries', 'yogurt',
    'green bean', 'sausage', 'pizza margherita', 'mushroom', 'tart',
    'white coffee', 'sunflower seeds', 'red bell pepper', 'asparagus', 'tartar sauce',
    'lye pretzel', 'pickled cucumber', 'vegetarian curry', 'lentil soup', 'vegetable salt cake',
    'heavy cream', 'chocolate cake', 'spaghetti', 'black olives', 'parmesan',
    'lambs ear salad', 'leaf salad', 'white cabbage', 'beetroot', 'grain bread',
    'raclette cheese', 'white bread', 'curds', 'quiche', 'beef',
    'taboule', 'eggplant', 'mozzarella', 'vegetable lasagne', 'mandarine',
    'french beans', 'spring roll', 'caprese salad', 'leaf spinach', 'white bread roll',
    'omelette', 'tuna', 'dark chocolate', 'savoury sauce', 'raisins',
    'black tea ice tea', 'kaki', 'smoothie', 'crepe', 'nuggets',
    'chili con carne', 'veggie burger', 'chinese cabbage', 'hamburger', 'pumpkin soup',
    'sushi', 'chestnuts', 'soya sauce', 'balsamic salad dressing', 'pasta twist',
    'bolognaise sauce', 'fajita bread', 'rice noodles vermicelli', 'whole wheat bread', 'onion',
    'garlic', 'vegetable pizza', 'beer', 'glucose drink', 'peanut',
    'green olives', 'wholemeal pasta', 'pesto sauce', 'couscous', 'toast',
    'water with lemon', 'espresso', 'braided white loaf', 'hazelnut chocolate spread', 'tomme',
    'hazelnut', 'peach', 'figs', 'pumpkin', 'swiss chard',
    'chicken curry', 'crunch muesli', 'biscuit', 'fresh cheese', 'vegetable mix with peas and carrots',
    'ice cream', 'dried meat', 'feta', 'praline', 'potato salad',
    'kohlrabi', 'alfa sprouts', 'Brussels sprouts', 'Gruyere', 'Bulgur',
    'Grapes', 'Chocolate egg', 'Cappuccino', 'Crisp bread', 'Black bread',
    'Rosti', 'Mango', 'Muesli', 'Spinach', 'Fish',
    'Risotto', 'Crisps', 'Pork', 'Pomegranate', 'Sweet corn',
    'Flakes', 'Greek salad', 'sesame seeds', 'bouillon', 'baked potato',
    'fennel', 'meat', 'bell pepper red stewed', 'nuts', 'breadcrumbs',
    'fondue', 'mushroom sauce', 'strawberries', 'plum pie', 'potatoes au gratin',
    'capers', 'wholemeal toast', 'red radish', 'fruit tart', 'kidney beans',
    'country fries', 'pasta linguini parpadelle tagliatelle', 'chicken strips', 'cookies', 'sun dried tomato',
    'bread ticino', 'semi hard cheese', 'porridge', 'juice', 'chocolate milk',
    'bread fruit', 'dates', 'pistachio', 'cream cheese', 'bread rye',
    'witloof chicory', 'goat cheese soft', 'grapefruit pomelo', 'blue cheese', 'guacamole',
    'cordon bleu', 'kefir', 'rocket', 'pizza ham mushrooms', 'fruit coulis',
    'plums', 'pizza ham', 'pineapple', 'seeds', 'focaccia',
    'milk beverage', 'coleslaw', 'chicken leg', 'cheesecake', 'chocolate croissant',
    'pumpkin seeds', 'artichoke', 'soft drink', 'apple pie', 'white bread',
    'pastry stick', 'tuna', 'meat pate', 'falafel', 'berries',
    'latte macchiato', 'melon', 'mixed seeds', 'celeriac', 'lemon',
    'chocolate cookies', 'birchermuesli no sugar', 'pine nuts', 'french pizza alsace', 'chocolate',
    'grits polenta', 'rose wine', 'cola drink', 'raspberries', 'chocolate roll',
    'lemon cake', 'gluten free bread', 'pearl onion', 'tzatziki', 'ham croissant',
    'corn crisps', 'green lentils', 'whole grain rice', 'cervelat', 'aperitif with alcohol',
    'apricots', 'lasagne meat', 'brioche', 'vegetable au gratin', 'basil',
    'almond butter', 'apricot pie', 'wholemeal rusk', 'conch pasta', 'pasta in butterfly form (farfalle)',
    'damson plum', 'shoots', 'coconut', 'banana cake', 'watermelon',
    'white asparagus', 'cherries', 'nectarine', 'small sauce-glass', 'white plate',
    'shallow bowl', 'caesar salad', 'parmesan dressing', 'egg cooked', 'toast croutons',
    'parmesan shavings', 'dressing', 'white plate without rim', 'soup of the day ratatouille cream', 'soup-bowl',
    'beer sauce', 'finger-shaped potato dumplings (schupfnudeln)', 'hioumi', 'natural yogurt', 'raspberry yogurt',
    'beans green', 'thyme', 'horn-shaped pasta (hoernli)', 'big sauce-glass', 'soup of the day artichoke',
    'onion sauce', 'glass fruitsalad-bowl', 'spanish tortilla', 'vanilla cream puffs', 'small quadratic plate-bowl',
    'basil pesto', 'quadratic dessert-plate', 'sausage and cheese salad', 'lollo bianco', 'house bread',
    'soup of the day potato', 'vegetarian bami goreng', 'lollo rosso', 'soup of the day broccoli cream', 'vegetable strips',
    'saffron herb sauce', 'vegetable strips saffron-herb sauce', 'pureed green balls', 'pureed food in a special shape', 'pureed meat slices',
    'pureed food in oval shape', 'pureed broccoli', 'pureed mashed potatoes', 'turmeric', 'sprout vegetables',
    'scallion', 'herbal rice', 'soup of the day curry cream', 'paneer', 'quinoa patties',
    'vegetables for quinoa patties', 'turkey ham', 'onion red', 'cranberry', 'barley risotto',
    'lemon panna cotta', 'parsley fritters', 'smoked trout', 'trout tartare', 'soup of the day yellow pea',
    'horseradish foam', 'vegetable lasagna', 'dill mashed potatoes', 'salmon cubes marinated', 'white wine sauce',
    'brown sauce', 'pureed food in pyramid shape', 'penne rigate', 'pureed balls', 'tagliatelle',
    'pureed cauliflower', 'raspberry', 'raspberry mousse in pyramid shape', 'bolognese', 'white sauce',
    'round raspberry mousse', 'slices', 'poultry stew', 'pureed salmon', 'pureed chicken thigh',
    'pureed fries', 'pureed sausage', 'plate with red rim', 'veggie swiss macaroni and cheese', 'poulet',
    'boiled meat salad seed oil', 'vegetable patch', 'boiled meat', 'currant sheet cake', 'bulgur sauce',
    'sliced seitan', 'oyster mushrooms', 'vegetables for green spelt risotto', 'green spelt risotto', 'cold chicken breast',
    'soup of the day carrot cream', 'curry dip', 'soup of the day lentil ginger', 'poulet cordon bleu', 'jus',
    'pilau rice', 'roasted cauliflower', 'sauce for sliced seitan', 'small plastic cup', 'overly soft thick brie cheese',
    'overly soft cottage cheese', 'meat cheese', 'lyonnaise potatoes', 'oversoaked sliced veal', 'overly soft cream cheese',
    'overly soft thin brie cheese', 'currants', 'soup of the day bell peppers', 'sliced quorn sauce zurich style', 'sliced quorn zurich style',
    'colorful vegetables from zuchetti peas carrots and beans', 'hash brown (roesti)', 'bread dumplings', 'sauce poultry ragout', 'banana organic',
    'lye croissant', 'lid on the ground', 'uncovered jug', 'jug covered with lid', 'mueesli',
    'large glass fruitsalad-bowl', 'milk roll', 'baked vegetables for mozzarella', 'wedges', 'chipolata sausage',
    'vegetables for chipolata sausage', 'pepper', 'rucola', 'oven vegetables', 'zuchetti', 'piccata mass',
    'vegetables for piccata', 'fried rice', 'beef meatballs', 'spicy vegetable ragout', 'turkey breast',
    'chili with vegetables', 'lenses brown', 'lenses', 'soggy bread without crust', 'big square plate',
    'soup of the day mushroom cream', 'oversoaked roast beef', 'oversoaked food in pyramid shape', 'oversoaked chia pudding', 'oversoaked mixed roast beef',
    'cheese sauce', ' swiss chard', 'oversoaked mixed chickpea curry', 'smoked sausage (landjaeger)', 'soup of the day leek cream',
    'vegetables for fregola', 'radish salad', 'fresh cheese praline', 'soup of the day banana-coconut', 'pickled vegetables',
    'deli meat cheese', 'turkey', 'cylindrical transparent shot-glass', 'ricotta tortellini', 'potato vegetable curry',
    'soup of the day sweetcorn', 'baked chickpea', 'milk coffee', 'cherry jam', 'coffee cup',
    'coffee plate', 'appenzeller cheese', 'paprika sauce', 'bramata', 'green spelt dumplings',
    'vegetable ragout for green spelt dumplings', 'chicken thigh steak', 'soup of the day tomatoes', 'vegetable salad for ham', 'country smoked ham',
    'lye rolls', 'antipasti vegetables', 'tagliatelle tomato pesto antipasti', 'soup of the day beetroot', 'soy yogurt dip',
    'bell pepper stew', 'pineapple-quark-mousse', 'quinoa salad', 'dried tomatoes', 'endives orange salad',
    'orange fillet', 'mascarpone', 'shiitake', 'red onion', 'little glass bowl',
    'vegetable salad for quinoa', 'halibut', 'asian dip', 'potato hash brown (roesti) with vegetables', 'oversoaked salmon fillet',
    'oversoaked chickpea puree', 'port wine pears rucola risotto', 'soup of the day barley', 'gorgonzola', 'creamy polenta medium',
    'beef patties in juicy sauce', 'merlot sauce', 'oversoaked perch fillet', 'oversoft food in crescent shape', 'rocket risotto',
    'oversoaked bell peppers', 'oversoakeboiled beef', 'oversoakeboiled polenta', 'oversoaked carrots', 'soft zuchetti',
    'veggie cervalat sausage', 'grisons barley soup', 'soup of the day cauliflower cream', 'herbal semolina slice', 'crispy vegetable roll',
    'soup of the day parmesan foam', 'gnocchi pan tofu', 'homemade fishburgers', 'soup of the day zucchetti', 'egg vinaigrette',
    'cheesy soft egg noodles (kaesespaetzle)', 'fennel salad for bowl', 'tree nut dressing', 'spelt marinated', 'feta marinated',
    'beetroot cooked', 'oversoaked smoked salmon', 'softened panna cotta', 'protein bowl', 'oversoaked fennel',
    'oversoaked couscous', 'oversoaked turkey plate', 'wild rice raw', 'homemade veggie burger', 'champignon organic',
    'pork steak', 'swiss macaroni and cheese', 'cantadou cheese', 'sliced beef', 'knot rolls',
    'spelt goulash', 'minced poultry patties', 'carbonara',)

# ALL_LABELS = ('chervil cream sauce', 'small sauce-glass', 'cream tart', 'white plate', 'chicken breast', 'tortellini', 'shallow bowl', 'cherry tomato', 'caesar salad', 'parmesan dressing', 'egg cooked', 'toast croutons', 'parmesan shavings', 'bacon', 'lettuce', 'dressing', 'colorful vegetable pan with soft egg noodles (spaetzle)', 'white plate without rim', 'capuns', 'soup of the day ratatouille cream', 'root vegetables', 'soup-bowl', 'beer sauce', 'finger-shaped potato dumplings (schupfnudeln)', 'sauerkraut', 'smoked pork neck', 'roasted cashew nuts', 'hioumi', 'raspberry quark', 'natural yogurt', 'strawberry yogurt', 'apricot yogurt', 'raspberry yogurt', 'beans green', 'cream sauce', 'vegetable bolognese', 'potatoes', 'thyme', 'horn-shaped pasta (hoernli)', 'caramel flan', 'big sauce-glass', 'fruit salad', 'soup of the day artichoke', 'veal sausage', 'onion sauce', 'glass fruitsalad-bowl', 'spanish tortilla', 'basil sauce', 'vanilla cream puffs', 'small quadratic plate-bowl', 'basil pesto', 'quadratic dessert-plate', 'radish', 'sausage and cheese salad', 'lollo bianco', 'house bread', 'soup of the day potato', 'vegetarian bami goreng', 'lollo rosso', 'soup of the day broccoli cream', 'spaghetti', 'vegetable strips', 'saffron herb sauce', 'vegetable strips saffron-herb sauce', 'pureed green balls', 'pureed food in a special shape', 'eggplant moussaka', 'pureed meat slices', 'pureed food in oval shape', 'pureed broccoli', 'pureed mashed potatoes', 'spinach tart', 'turmeric', 'pita bread', 'trout fillet', 'spinach', 'rice', 'sauce', 'sprout vegetables', 'scallion', 'herbal rice', 'soup of the day curry cream', 'lentil ragout', 'paneer', 'quinoa patties', 'vegetables for quinoa patties', 'toast', 'turkey ham', 'pineapple', 'gruyere cheese', 'onion red', 'cranberry', 'barley risotto', 'salad leaves', 'cheese tart', 'zucchetti', 'lemon panna cotta', 'parsley fritters', 'lemon', 'capers', 'smoked trout', 'trout tartare', 'soup of the day yellow pea', 'horseradish foam', 'butter', 'vegetable lasagna', 'applesauce', 'broccoli', 'dill mashed potatoes', 'salmon cubes marinated', 'white wine sauce', 'brown sauce', 'pureed food in pyramid shape', 'penne rigate', 'pureed polenta', 'pureed balls', 'tagliatelle', 'pureed cauliflower', 'raspberry', 'raspberry mousse in pyramid shape', 'bolognese', 'white sauce', 'bell pepper sauce', 'chocolate mousse', 'round raspberry mousse', 'romanesco', 'slices', 'poultry stew', 'pureed salmon', 'pureed chicken thigh', 'pureed fries', 'pureed sausage', 'penne', 'plate with red rim', 'soft egg noodles (spaetzle)', 'veggie swiss macaroni and cheese', 'poulet', 'pasta', 'boiled meat salad seed oil', 'vegetable strudel', 'vegetable patch', 'herb quark dip', 'vegetables for boiled meat salad', 'plum tart', 'boiled meat', 'gnocchi seitan pan', 'lamb stew', 'bulgur', 'carrots', 'currant sheet cake', 'bulgur sauce', 'gnocchi', 'sliced seitan', 'oyster mushrooms', 'vegetable salad with white beans', 'vegetables for green spelt risotto', 'green spelt risotto', 'rye bread', 'apple tart', 'bouillon', 'cold chicken breast', 'cocktail sauce', 'soup of the day carrot cream', 'curry dip', 'soup of the day lentil ginger', 'poulet cordon bleu', 'jus', 'pilau rice', 'sugar peas', 'roasted cauliflower', 'sauce for sliced seitan', 'boiled potatoes', 'semolina porridge', 'cherry compote', 'cinnamon sugar', 'small plastic cup', 'oversoaked sliced chicken', 'overly soft thick brie cheese', 'overly soft cottage cheese', 'meat cheese', 'mustard sauce', 'lyonnaise potatoes', 'oversoaked sliced veal', 'overly soft cream cheese', 'overly soft thin brie cheese', 'currants', 'soup of the day bell peppers', 'sliced quorn sauce zurich style', 'sliced quorn zurich style', 'colorful vegetables from zuchetti peas carrots and beans', 'hash brown (roesti)', 'bread dumplings', 'sauce poultry ragout', 'sliced quorn', 'vegetables', 'banana organic', 'croissant', 'lye croissant', 'coffee', 'lid on the ground', 'uncovered jug', 'jug covered with lid', 'orange juice', 'multigrain roll', 'scrambled eggs', 'milk', 'mueesli', 'large glass fruitsalad-bowl', 'milk roll', 'mozzarella', 'baked vegetables for mozzarella', 'wedges', 'chipolata sausage', 'vegetables for chipolata sausage', 'pepper', 'rucola', 'walnut', 'mozzarella salad', 'oven vegetables', 'zuchetti', 'eggplant', 'piccata mass', 'bramata slice', 'vegetables for piccata', 'cream', 'chocolate cake', 'fried rice', 'beef meatballs', 'spicy vegetable ragout', 'olives', 'cucumber', 'tomato', 'turkey breast', 'carrot', 'chili with vegetables', 'lenses brown', 'lenses', 'pear', 'apple', 'apricot tart', 'cheese crêpe', 'meatloaf', 'peas', 'mashed potatoes', 'soggy bread without crust', 'big square plate', 'oversoaked polenta', 'soup of the day mushroom cream', 'cognac sauce', 'grated cheese', 'oversoaked chickpea curry', 'oversoaked roast beef', 'oversoaked food in pyramid shape', 'swiss chard vegetable ragout', 'oversoaked chia pudding', 'oversoaked mixed roast beef', 'cheese sauce', ' swiss chard', 'oversoaked mixed chickpea curry', 'sardinian fregola', 'smoked sausage (landjaeger)', 'soup of the day leek cream', 'vegetables for fregola', 'mustard', 'radish salad', 'fresh cheese praline', 'pickled cucumber', 'soup of the day banana-coconut', 'bean cassoulet', 'salami', 'pickled vegetables', 'deli meat cheese', 'turkey', 'sour cream', 'cylindrical transparent shot-glass', 'ricotta tortellini', 'potato vegetable curry', 'soup of the day sweetcorn', 'baked chickpea', 'tomato sauce', 'milk coffee', 'cherry jam', 'spreadable cheese', 'coffee cup', 'coffee plate', 'coffee yogurt', 'tilster cheese', 'brie cheese', 'margarine', 'appenzeller cheese', 'paprika sauce', 'bramata', 'green spelt dumplings', 'vegetable ragout for green spelt dumplings', 'chicken thigh steak', 'country cuts', 'veggie crispy bites', 'soup of the day tomatoes', 'vegetable salad for ham', 'country smoked ham', 'lye rolls', 'antipasti vegetables', 'tagliatelle tomato pesto antipasti', 'soup of the day beetroot', 'soy yogurt dip', 'vegan meatballs', 'vegetables for meatballs', 'yellow pea puree', 'boiled beef', 'horseradish bouillon', 'beef lasagna', 'eggplant cordon bleu', 'saffron risotto', 'bell pepper stew', 'pineapple-quark-mousse', 'quinoa salad', 'dried tomatoes', 'endives orange salad', 'orange fillet', 'cashew nuts', 'mascarpone', 'shiitake', 'red onion', 'risotto', 'vegetable salad', 'quinoa', 'little glass bowl', 'vegetable salad for quinoa', 'thai glass noodle salad', 'cheese ravioli', 'fruit quark', 'halibut', 'hummus', 'vegetables for halibut', 'asian dip', 'potato hash brown (roesti) with vegetables', 'oversoaked salmon fillet', 'oversoaked chickpea puree', 'bell pepper', 'port wine pears rucola risotto', 'soup of the day barley', 'gorgonzola', 'creamy polenta medium', 'beef patties in juicy sauce', 'merlot sauce', 'oversoaked perch fillet', 'oversoft food in crescent shape', 'rocket risotto', 'oversoaked bell peppers', 'oversoakeboiled beef', 'oversoakeboiled polenta', 'oversoaked carrots', 'polenta', 'soft zuchetti', 'potato herb patties', 'veggie cervalat sausage', 'colorful vegetables for veggie cervalat sausage', 'grisons barley soup', 'carrot puree', 'cod', 'ratatouille', 'soup of the day cauliflower cream', 'herbal semolina slice', 'crispy vegetable roll', 'rice noodle salad', 'soup of the day parmesan foam', 'creamed spinach', 'gnocchi pan tofu', 'homemade fishburgers', 'black bean puree', 'soup of the day zucchetti', 'roast beef', 'pizokel vegetable gratin', 'egg vinaigrette', 'cheesy soft egg noodles (kaesespaetzle)', 'cabbage salad', 'fennel salad for bowl', 'tree nut dressing', 'spelt marinated', 'feta marinated', 'beetroot cooked', 'iceberg lettuce', 'oversoaked smoked salmon', 'softened panna cotta', 'protein bowl', 'oversoaked fennel', 'oversoaked couscous', 'oversoaked turkey plate', 'lemon roulade', 'wild rice raw', 'homemade veggie burger', 'spicy tomato vegetable sauce', 'champignon organic', 'pork steak', 'gruyère', 'swiss macaroni and cheese', 'cantadou cheese', 'white bean puree', 'tilapia', 'burrito', 'pernod sauce', 'diced tomatoes', 'sliced beef', 'knot rolls', 'spelt goulash', 'french dressing', 'beans', 'salad', 'minced poultry patties', 'tofu', 'carbonara', 'carbonara tofu', 'rosemary sauce')

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
        # Define probabilities for choosing n wrong labels
        probabilities = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.1, 5: 0.1}
        
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
