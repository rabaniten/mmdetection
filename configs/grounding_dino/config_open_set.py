# # Local training and inference
# LOAD_FROM = '/root/Sofia/Genioos/sofia_thesis_project/detection_models/grounding_dino/trained_models/epoch_40.pth'

# RESUME = False

# ANN_FILE_TRAINING = '/root/Sofia/Genioos/sofia_thesis_project/instance_segmentation_models/out_dirs/bbox_and_mask_annotations_obtained_from_gt_and_SAM_stadtpital_waid_train_data_coco_format_corrected_no_crowds_removed_small_masks.json'
# ANN_FILE_VALIDATION = '/root/Sofia/Genioos/sofia_thesis_project/instance_segmentation_models/out_dirs/bbox_and_mask_annotations_obtained_from_gt_and_SAM_stadtpital_waid_val_data_coco_format_corrected_no_crowds_removed_small_masks.json'

# DATA_PREFIX_TRAIN = dict(img= '/root/Sofia/Genioos/data/Stadtspital-Waid/annotated_data_for_ml_model/training_and_val_data/coco_training/images/')
# DATA_PREFIX_VAL = dict(img='/root/Sofia/Genioos/data/Stadtspital-Waid/annotated_data_for_ml_model/training_and_val_data/coco_validation/images/')

# BATCH_SIZE_TRAIN = 1
# BATCH_SIZE_VAL = 1

# NUM_WORKER_TRAIN = 2
# NUM_WORKER_VAL = 2


# Training and inference in custom docker
LOAD_FROM = "/opt/ml/code/pretrained_models/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
# LOAD_FROM = '/opt/ml/code/pretrained_models/epoch_40.pth'

RESUME = False  # Enable resume to continue training

ANN_FILE_TRAINING = "/opt/ml/input/data/train/annotations/instances_train.json"
ANN_FILE_VALIDATION = "/opt/ml/input/data/validation/annotations/instances_val.json"

DATA_PREFIX_TRAIN = dict(img="/opt/ml/input/data/train/images/")
DATA_PREFIX_VAL = dict(img="/opt/ml/input/data/validation/images/")

BATCH_SIZE_TRAIN = 1
BATCH_SIZE_VAL = 1

NUM_WORKER_TRAIN = 32
NUM_WORKER_VAL = 32

MAX_EPOCHS = 50  # Train for 20 more epochs (total 50)


CLASSES = (
    "coffee cup",
    "coffee jar",
    "glass",
    "high bowl",
    "normal bowl",
    "other tableware",
    "plate large",
    "plate normal",
    "plate small",
    "small tableware item",
    "square bowl",
    "wide bowl",
    "tableware",
)


# CLASSES = ('Other',
#     'banana', 'caramel flan', 'lamb stew', 'carrots', 'bread',
#     'soup', 'chanterelle parsley risotto', 'panna cotta', 'french salad dressing', 'corn',
#     'pomegranate', 'chocolate ice cream', 'applesauce', 'bulgur', 'gnocchi seitan pan',
#     'whipped cream', 'Plum muffin', 'grated cheese', 'lettuce', 'mashed potatoes',
#     'mashed potato', 'oversoaked beef roast', 'peas', 'carrot', 'chipolata',
#     'potato wedges', 'italian sauce', 'turnip cabbage', 'vanilla ice cream', 'apple',
#     'fruit salad', 'mozzarella salad', 'zucchini', 'pureed carrot', 'pureed beef',
#     'cream sauce', 'lemon slice', 'caper butter sauce', 'potatoes', 'sole',
#     'beans', 'cherry tomato', 'mixed salad', 'meatloaf', 'cognac sauce',
#     'herb cream sauce', 'eggplant moussaka', 'strawberry yogurt', 'hollandaise sauce', 'legume salad',
#     'macaroni', 'tomato sauce', 'vanilla porridge', 'apricot yogurt', 'cream slice',
#     'butter', 'gravy', 'bircher muesli', 'boiled egg', 'Cherry Tomato',
#     'vegetables for boiled meat salad', 'shrimps', 'walnut', 'cheese spaetzle', 'cabbage salad',
#     'baby lettuce', 'radish', 'orange', 'cocktail sauce', 'parsley',
#     'fried onions', 'vegetable strudel', 'herb quark dip', 'chocolate yogurt', 'mixed carrot and peas',
#     'rice', 'coffee yogurt', 'pureed bratwurst', 'vanilla cream', 'herb rice',
#     'trout fillet', 'spinach leaf', 'thin chocolate decoration', 'salmon cubes', 'oversoaked millet slice',
#     'zucchetti', 'oversoaked broccoli', 'oversoaked sliced chicken', 'oversoaked bean', 'rustico croissant',
#     'multigrain roll', 'honey', 'protein drink', 'cheese crepe', 'swiss chard vegetable ragout',
#     'mustard sauce', 'mint', 'moustard sauce', 'baked meatloaf', 'lingonberry compote',
#     'Quail Breast', 'Cranberries', 'Celery Salad', 'jam', 'bread roll',
#     'croissant', 'milk', 'chocolate powder bag', 'sliced veal', 'romanesco',
#     'oversoaked zucchetti', 'polenta', 'broccoli', 'beef braised slice', 'tiramisu slice',
#     'merlot jus', 'thin chocolate', 'balsamic sauce', 'strawberry ice cream', 'juniper jus',
#     'veal cheak', 'fregola', 'red cabbage', 'red chicory', 'cooked beetroot',
#     'orange segments', 'walnuts', 'peeled carrot', 'iceberg lettuce', 'walnut dressing',
#     'chocolate mousse', 'grapes', 'cheese', 'salad', 'carrot appetizer',
#     'tilapia', 'pernod sauce', 'white bean puree', 'vegetable', 'nut cake',
#     'chopped herbs', 'tiramisu', 'boiled potatoes', 'cheese spread', 'minced meat',
#     'oversoaked chicken strips with sauce', 'penne pasta', 'scrambled eggs', 'creamed spinach', 'rice noodle salad',
#     'chili pepper', 'peanuts', 'lactose-free milk', 'Jam sandwich cookie', 'chicken breast',
#     'yellow pea puree', 'plain yogurt', 'apple tart', 'cream', 'veggie crispy bites',
#     'oversoaked polenta', 'Pork Steak', 'soft egg noodles (spaetzle)', 'Root Vegetables', 'vegetables for meatballs',
#     'vegan meatballs', 'Tagliatelle', 'sauce', 'vegetables', 'cheese ravioli',
#     'lemon sorbet', 'cashew nuts', 'smoked sausage', 'mixed vegetables', 'sardinian fregola',
#     'sliced quorn', 'basil sauce', 'beef lasagna', 'glazed carrots', 'beef roast',
#     'special bean', 'saffron risotto', 'bell pepper sauce', 'eggplant cordon bleu', 'peperonata',
#     'oversoaked creamed spinach', 'white wine', 'cod', 'spinach tart', 'breaded poultry meatball',
#     'rosemary sauce', 'cauliflower', 'yeast roll', 'oversoaked bolognese', 'chickpea puree',
#     'bread without crust', 'plum', 'penne', 'diced tomatoes', 'carbonara tofu',
#     'kiwi', 'pizokel vegetable gratin', 'plum tart', 'rusk', 'coffee',
#     'emmental cheese', 'margarine', 'hash brown (roesti) with cheese', 'orange segment', 'oversoaked celery',
#     'ham sandwich', 'sweet potato', 'soft cheese', 'spreadable cheese', 'fruit jelly',
#     'quince jelly', 'cottage cheese', 'wild rice', 'spicy tomato vegetable sauce', 'vegetarian burger',
#     'mushrooms', 'sea bass', 'bok choy', 'oriental rice', 'boiled beef',
#     'root vegetables', 'broth', 'plum crumble', 'mayonnaise', 'greyerzer cheese',
#     'apricose quark', 'veggie sausage', 'potato herb patties', 'colorful vegetables for veggie cervalat sausage', 'swedish cake',
#     'golden berry', 'oversoaked boiled potatoes', 'chicken thigh', 'bramata slice', 'cream herb sauce',
#     'vegetable ragout', 'spelt dumplings', 'brownie', 'fruit quark', 'gran padrone',
#     'thai glass noodle salad', 'knot bread rolls', 'red wine', 'leek', 'colorful vegetable pan with soft egg noodles (spaetzle)',
#     'Cashew Nuts', 'Halloumi', 'sauteed tomato', 'tofu', 'raspberry quark',
#     'mac and cheese', 'gruyere cheese', 'coffee cream', 'spinach', 'sliced turkey breast',
#     'tomato cream sauce', 'tortellini', 'oversoaked carrot', 'beef tartare', 'raspberry yogurt lactose-free',
#     'quinoa', 'vegetable salad', 'apricot tart', 'oversoaked salmon', 'endive orange salad',
#     'risotto', 'fish burger', 'black bean puree', 'mashed black beans', 'apple cookie',
#     'chives', 'bread dumpling', 'poultry ragout', 'sugar peas', 'oversoaked scrambled egg',
#     'potato', 'Bread', 'scrambled egg', 'sesame tofu', 'rosti',
#     'quorn strips in cream sauce', 'oversoaked cauliflower', 'pureed polenta', 'apple juice', 'cream tart',
#     'chervil cream sauce', 'croutons', 'bacon', 'salad dressing', 'gran padano',
#     'orange juice', 'strawberry quark', 'tilster cheese', 'Chicken', 'pasta',
#     'grape', 'vegetable bolognese', 'cinnamon sugar', 'cherry compote', 'semolina porridge',
#     'ketchup', 'buttered pretzel', 'beet ginger salad', 'salmon', 'cholocate drink',
#     'lemon roulade', 'pilaw rice', 'sour cream', 'burrito', 'country cuts',
#     'green beans', 'vegan  nuggets', 'pilaf rice', 'pureed omelette', 'mashed semolina',
#     'fruit', 'oversoaked bell pepper', 'oversoaked chickpea curry', 'Aufschnittteller VVG', 'bean cassoulet',
#     'salad leaves', 'Choernlibroetli', 'Aufschnittteller', 'Bean Cassoulet', 'cucumber',
#     'salami', 'turkey cold cut', 'boiled eggs', 'roast beef', 'vinaigrette',
#     'olives', 'Halibut', 'vegetables for halibut', 'pasta with tomato sauce', 'hummus',
#     'whole grain rice cake', 'tea', 'horseradish bouillon', 'salted potatoes', 'root vegetable',
#     'lamb loin', 'special jus', 'mustard', 'bag of Ovaltine', 'cheese sandwich',
#     'bell pepper', 'cod with herbs', 'ratatouille', 'herb semolina slice', 'carrot puree',
#     'springroll', 'halloumi', 'roasted cashew nuts', 'colorful spaetzle', 'cheese tart',
#     'rye bread', 'radishes', 'brie cheese', 'gruyere', 'crispy fried onions',
#     'chickpea triangles', 'vegetable curry', 'red pepperoncini', 'spring onion', 'oversoaked cheese plate',
#     'gnocchi', 'celery', '"Salade nicoise"', 'minced beef sauce', 'vegetable salad with white beans',
#     'chicken breast slices', 'curry sauce', 'bami goreng', 'finger-shaped potato dumplings', 'smoked pork neck',
#     'sauerkraut', 'mashed pasta', 'pureed chicken', 'Red Onion', 'Cranberry',
#     'hawaiian toast', 'Seitan Strips', 'Sauce', 'Spanish Tortilla', 'veal sausage',
#     'vegetable salad with feta', 'mashed peas', 'lentil ragout', 'pita bread', 'sprout vegetable',
#     'muffin', 'soft bircher muesli', 'quark', 'sweet and sour carrot', 'salami sandwich',
#     'mushroom cream sauce', 'capuns', 'Beetroot', 'Wholemeal bread', 'Jam',
#     'Water', 'Banana', 'Soft cheese', 'Raw ham', 'Hard cheese',
#     'Cottage cheese', 'Coffee', 'Mixed fruit', 'Pancake', 'Tea',
#     'Smoked salmon', 'Avocado', 'Spring onion', 'Ristretto', 'Ham',
#     'Egg', 'french fries', 'chicken', 'tomato', 'shrimp',
#     'chickpeas', 'french dressing', 'horn shaped pasta', 'pear', 'cashew',
#     'almonds', 'lentil', 'peanut butter', 'blueberries', 'yogurt',
#     'green bean', 'sausage', 'pizza margherita', 'mushroom', 'tart',
#     'white coffee', 'sunflower seeds', 'red bell pepper', 'asparagus', 'tartar sauce',
#     'lye pretzel', 'pickled cucumber', 'vegetarian curry', 'lentil soup', 'vegetable salt cake',
#     'heavy cream', 'chocolate cake', 'spaghetti', 'black olives', 'parmesan',
#     'lambs ear salad', 'leaf salad', 'white cabbage', 'beetroot', 'grain bread',
#     'raclette cheese', 'white bread', 'curds', 'quiche', 'beef',
#     'taboule', 'eggplant', 'mozzarella', 'vegetable lasagne', 'mandarine',
#     'french beans', 'spring roll', 'caprese salad', 'leaf spinach', 'white bread roll',
#     'omelette', 'tuna', 'dark chocolate', 'savoury sauce', 'raisins',
#     'black tea ice tea', 'kaki', 'smoothie', 'crepe', 'nuggets',
#     'chili con carne', 'veggie burger', 'chinese cabbage', 'hamburger', 'pumpkin soup',
#     'sushi', 'chestnuts', 'soya sauce', 'balsamic salad dressing', 'pasta twist',
#     'bolognaise sauce', 'fajita bread', 'rice noodles vermicelli', 'whole wheat bread', 'onion',
#     'garlic', 'vegetable pizza', 'beer', 'glucose drink', 'peanut',
#     'green olives', 'wholemeal pasta', 'pesto sauce', 'couscous', 'toast',
#     'water with lemon', 'espresso', 'braided white loaf', 'hazelnut chocolate spread', 'tomme',
#     'hazelnut', 'peach', 'figs', 'pumpkin', 'swiss chard',
#     'chicken curry', 'crunch muesli', 'biscuit', 'fresh cheese', 'vegetable mix with peas and carrots',
#     'ice cream', 'dried meat', 'feta', 'praline', 'potato salad',
#     'kohlrabi', 'alfa sprouts', 'Brussels sprouts', 'Gruyere', 'Bulgur',
#     'Grapes', 'Chocolate egg', 'Cappuccino', 'Crisp bread', 'Black bread',
#     'Rosti', 'Mango', 'Muesli', 'Spinach', 'Fish',
#     'Risotto', 'Crisps', 'Pork', 'Pomegranate', 'Sweet corn',
#     'Flakes', 'Greek salad', 'sesame seeds', 'bouillon', 'baked potato',
#     'fennel', 'meat', 'bell pepper red stewed', 'nuts', 'breadcrumbs',
#     'fondue', 'mushroom sauce', 'strawberries', 'plum pie', 'potatoes au gratin',
#     'capers', 'wholemeal toast', 'red radish', 'fruit tart', 'kidney beans',
#     'country fries', 'pasta linguini parpadelle tagliatelle', 'chicken strips', 'cookies', 'sun dried tomato',
#     'bread ticino', 'semi hard cheese', 'porridge', 'juice', 'chocolate milk',
#     'bread fruit', 'dates', 'pistachio', 'cream cheese', 'bread rye',
#     'witloof chicory', 'goat cheese soft', 'grapefruit pomelo', 'blue cheese', 'guacamole',
#     'cordon bleu', 'kefir', 'rocket', 'pizza ham mushrooms', 'fruit coulis',
#     'plums', 'pizza ham', 'pineapple', 'seeds', 'focaccia',
#     'milk beverage', 'coleslaw', 'chicken leg', 'cheesecake', 'chocolate croissant',
#     'pumpkin seeds', 'artichoke', 'soft drink', 'apple pie', 'white bread',
#     'pastry stick', 'tuna', 'meat pate', 'falafel', 'berries',
#     'latte macchiato', 'melon', 'mixed seeds', 'celeriac', 'lemon',
#     'chocolate cookies', 'birchermuesli no sugar', 'pine nuts', 'french pizza alsace', 'chocolate',
#     'grits polenta', 'rose wine', 'cola drink', 'raspberries', 'chocolate roll',
#     'lemon cake', 'gluten free bread', 'pearl onion', 'tzatziki', 'ham croissant',
#     'corn crisps', 'green lentils', 'whole grain rice', 'cervelat', 'aperitif with alcohol',
#     'apricots', 'lasagne meat', 'brioche', 'vegetable au gratin', 'basil',
#     'almond butter', 'apricot pie', 'wholemeal rusk', 'conch pasta', 'pasta in butterfly form (farfalle)',
#     'damson plum', 'shoots', 'coconut', 'banana cake', 'watermelon',
#     'white asparagus', 'cherries', 'nectarine', 'small sauce-glass', 'white plate',
#     'shallow bowl', 'caesar salad', 'parmesan dressing', 'egg cooked', 'toast croutons',
#     'parmesan shavings', 'dressing', 'white plate without rim', 'soup of the day ratatouille cream', 'soup-bowl',
#     'beer sauce', 'finger-shaped potato dumplings (schupfnudeln)', 'hioumi', 'natural yogurt', 'raspberry yogurt',
#     'beans green', 'thyme', 'horn-shaped pasta (hoernli)', 'big sauce-glass', 'soup of the day artichoke',
#     'onion sauce', 'glass fruitsalad-bowl', 'spanish tortilla', 'vanilla cream puffs', 'small quadratic plate-bowl',
#     'basil pesto', 'quadratic dessert-plate', 'sausage and cheese salad', 'lollo bianco', 'house bread',
#     'soup of the day potato', 'vegetarian bami goreng', 'lollo rosso', 'soup of the day broccoli cream', 'vegetable strips',
#     'saffron herb sauce', 'vegetable strips saffron-herb sauce', 'pureed green balls', 'pureed food in a special shape', 'pureed meat slices',
#     'pureed food in oval shape', 'pureed broccoli', 'pureed mashed potatoes', 'turmeric', 'sprout vegetables',
#     'scallion', 'herbal rice', 'soup of the day curry cream', 'paneer', 'quinoa patties',
#     'vegetables for quinoa patties', 'turkey ham', 'onion red', 'cranberry', 'barley risotto',
#     'lemon panna cotta', 'parsley fritters', 'smoked trout', 'trout tartare', 'soup of the day yellow pea',
#     'horseradish foam', 'vegetable lasagna', 'dill mashed potatoes', 'salmon cubes marinated', 'white wine sauce',
#     'brown sauce', 'pureed food in pyramid shape', 'penne rigate', 'pureed balls', 'tagliatelle',
#     'pureed cauliflower', 'raspberry', 'raspberry mousse in pyramid shape', 'bolognese', 'white sauce',
#     'round raspberry mousse', 'slices', 'poultry stew', 'pureed salmon', 'pureed chicken thigh',
#     'pureed fries', 'pureed sausage', 'plate with red rim', 'veggie swiss macaroni and cheese', 'poulet',
#     'boiled meat salad seed oil', 'vegetable patch', 'boiled meat', 'currant sheet cake', 'bulgur sauce',
#     'sliced seitan', 'oyster mushrooms', 'vegetables for green spelt risotto', 'green spelt risotto', 'cold chicken breast',
#     'soup of the day carrot cream', 'curry dip', 'soup of the day lentil ginger', 'poulet cordon bleu', 'jus',
#     'pilau rice', 'roasted cauliflower', 'sauce for sliced seitan', 'small plastic cup', 'overly soft thick brie cheese',
#     'overly soft cottage cheese', 'meat cheese', 'lyonnaise potatoes', 'oversoaked sliced veal', 'overly soft cream cheese',
#     'overly soft thin brie cheese', 'currants', 'soup of the day bell peppers', 'sliced quorn sauce zurich style', 'sliced quorn zurich style',
#     'colorful vegetables from zuchetti peas carrots and beans', 'hash brown (roesti)', 'bread dumplings', 'sauce poultry ragout', 'banana organic',
#     'lye croissant', 'lid on the ground', 'uncovered jug', 'jug covered with lid', 'mueesli',
#     'large glass fruitsalad-bowl', 'milk roll', 'baked vegetables for mozzarella', 'wedges', 'chipolata sausage',
#     'vegetables for chipolata sausage', 'pepper', 'rucola', 'oven vegetables', 'zuchetti', 'piccata mass',
#     'vegetables for piccata', 'fried rice', 'beef meatballs', 'spicy vegetable ragout', 'turkey breast',
#     'chili with vegetables', 'lenses brown', 'lenses', 'soggy bread without crust', 'big square plate',
#     'soup of the day mushroom cream', 'oversoaked roast beef', 'oversoaked food in pyramid shape', 'oversoaked chia pudding', 'oversoaked mixed roast beef',
#     'cheese sauce', ' swiss chard', 'oversoaked mixed chickpea curry', 'smoked sausage (landjaeger)', 'soup of the day leek cream',
#     'vegetables for fregola', 'radish salad', 'fresh cheese praline', 'soup of the day banana-coconut', 'pickled vegetables',
#     'deli meat cheese', 'turkey', 'cylindrical transparent shot-glass', 'ricotta tortellini', 'potato vegetable curry',
#     'soup of the day sweetcorn', 'baked chickpea', 'milk coffee', 'cherry jam', 'coffee cup',
#     'coffee plate', 'appenzeller cheese', 'paprika sauce', 'bramata', 'green spelt dumplings',
#     'vegetable ragout for green spelt dumplings', 'chicken thigh steak', 'soup of the day tomatoes', 'vegetable salad for ham', 'country smoked ham',
#     'lye rolls', 'antipasti vegetables', 'tagliatelle tomato pesto antipasti', 'soup of the day beetroot', 'soy yogurt dip',
#     'bell pepper stew', 'pineapple-quark-mousse', 'quinoa salad', 'dried tomatoes', 'endives orange salad',
#     'orange fillet', 'mascarpone', 'shiitake', 'red onion', 'little glass bowl',
#     'vegetable salad for quinoa', 'halibut', 'asian dip', 'potato hash brown (roesti) with vegetables', 'oversoaked salmon fillet',
#     'oversoaked chickpea puree', 'port wine pears rucola risotto', 'soup of the day barley', 'gorgonzola', 'creamy polenta medium',
#     'beef patties in juicy sauce', 'merlot sauce', 'oversoaked perch fillet', 'oversoft food in crescent shape', 'rocket risotto',
#     'oversoaked bell peppers', 'oversoakeboiled beef', 'oversoakeboiled polenta', 'oversoaked carrots', 'soft zuchetti',
#     'veggie cervalat sausage', 'grisons barley soup', 'soup of the day cauliflower cream', 'herbal semolina slice', 'crispy vegetable roll',
#     'soup of the day parmesan foam', 'gnocchi pan tofu', 'homemade fishburgers', 'soup of the day zucchetti', 'egg vinaigrette',
#     'cheesy soft egg noodles (kaesespaetzle)', 'fennel salad for bowl', 'tree nut dressing', 'spelt marinated', 'feta marinated',
#     'beetroot cooked', 'oversoaked smoked salmon', 'softened panna cotta', 'protein bowl', 'oversoaked fennel',
#     'oversoaked couscous', 'oversoaked turkey plate', 'wild rice raw', 'homemade veggie burger', 'champignon organic',
#     'pork steak', 'swiss macaroni and cheese', 'cantadou cheese', 'sliced beef', 'knot rolls',
#     'spelt goulash', 'minced poultry patties', 'carbonara',)


# CLASSES = ('chervil cream sauce', 'small sauce-glass', 'cream tart', 'white plate', 'chicken breast', 'tortellini', 'shallow bowl', 'cherry tomato', 'caesar salad', 'parmesan dressing', 'egg cooked', 'toast croutons', 'parmesan shavings', 'bacon', 'lettuce', 'dressing', 'colorful vegetable pan with soft egg noodles (spaetzle)', 'white plate without rim', 'capuns', 'soup of the day ratatouille cream', 'root vegetables', 'soup-bowl', 'beer sauce', 'finger-shaped potato dumplings (schupfnudeln)', 'sauerkraut', 'smoked pork neck', 'roasted cashew nuts', 'hioumi', 'raspberry quark', 'natural yogurt', 'strawberry yogurt', 'apricot yogurt', 'raspberry yogurt', 'beans green', 'cream sauce', 'vegetable bolognese', 'potatoes', 'thyme', 'horn-shaped pasta (hoernli)', 'caramel flan', 'big sauce-glass', 'fruit salad', 'soup of the day artichoke', 'veal sausage', 'onion sauce', 'glass fruitsalad-bowl', 'spanish tortilla', 'basil sauce', 'vanilla cream puffs', 'small quadratic plate-bowl', 'basil pesto', 'quadratic dessert-plate', 'radish', 'sausage and cheese salad', 'lollo bianco', 'house bread', 'soup of the day potato', 'vegetarian bami goreng', 'lollo rosso', 'soup of the day broccoli cream', 'spaghetti', 'vegetable strips', 'saffron herb sauce', 'vegetable strips saffron-herb sauce', 'pureed green balls', 'pureed food in a special shape', 'eggplant moussaka', 'pureed meat slices', 'pureed food in oval shape', 'pureed broccoli', 'pureed mashed potatoes', 'spinach tart', 'turmeric', 'pita bread', 'trout fillet', 'spinach', 'rice', 'sauce', 'sprout vegetables', 'scallion', 'herbal rice', 'soup of the day curry cream', 'lentil ragout', 'paneer', 'quinoa patties', 'vegetables for quinoa patties', 'toast', 'turkey ham', 'pineapple', 'gruyere cheese', 'onion red', 'cranberry', 'barley risotto', 'salad leaves', 'cheese tart', 'zucchetti', 'lemon panna cotta', 'parsley fritters', 'lemon', 'capers', 'smoked trout', 'trout tartare', 'soup of the day yellow pea', 'horseradish foam', 'butter', 'vegetable lasagna', 'applesauce', 'broccoli', 'dill mashed potatoes', 'salmon cubes marinated', 'white wine sauce', 'brown sauce', 'pureed food in pyramid shape', 'penne rigate', 'pureed polenta', 'pureed balls', 'tagliatelle', 'pureed cauliflower', 'raspberry', 'raspberry mousse in pyramid shape', 'bolognese', 'white sauce', 'bell pepper sauce', 'chocolate mousse', 'round raspberry mousse', 'romanesco', 'slices', 'poultry stew', 'pureed salmon', 'pureed chicken thigh', 'pureed fries', 'pureed sausage', 'penne', 'plate with red rim', 'soft egg noodles (spaetzle)', 'veggie swiss macaroni and cheese', 'poulet', 'pasta', 'boiled meat salad seed oil', 'vegetable strudel', 'vegetable patch', 'herb quark dip', 'vegetables for boiled meat salad', 'plum tart', 'boiled meat', 'gnocchi seitan pan', 'lamb stew', 'bulgur', 'carrots', 'currant sheet cake', 'bulgur sauce', 'gnocchi', 'sliced seitan', 'oyster mushrooms', 'vegetable salad with white beans', 'vegetables for green spelt risotto', 'green spelt risotto', 'rye bread', 'apple tart', 'bouillon', 'cold chicken breast', 'cocktail sauce', 'soup of the day carrot cream', 'curry dip', 'soup of the day lentil ginger', 'poulet cordon bleu', 'jus', 'pilau rice', 'sugar peas', 'roasted cauliflower', 'sauce for sliced seitan', 'boiled potatoes', 'semolina porridge', 'cherry compote', 'cinnamon sugar', 'small plastic cup', 'oversoaked sliced chicken', 'overly soft thick brie cheese', 'overly soft cottage cheese', 'meat cheese', 'mustard sauce', 'lyonnaise potatoes', 'oversoaked sliced veal', 'overly soft cream cheese', 'overly soft thin brie cheese', 'currants', 'soup of the day bell peppers', 'sliced quorn sauce zurich style', 'sliced quorn zurich style', 'colorful vegetables from zuchetti peas carrots and beans', 'hash brown (roesti)', 'bread dumplings', 'sauce poultry ragout', 'sliced quorn', 'vegetables', 'banana organic', 'croissant', 'lye croissant', 'coffee', 'lid on the ground', 'uncovered jug', 'jug covered with lid', 'orange juice', 'multigrain roll', 'scrambled eggs', 'milk', 'mueesli', 'large glass fruitsalad-bowl', 'milk roll', 'mozzarella', 'baked vegetables for mozzarella', 'wedges', 'chipolata sausage', 'vegetables for chipolata sausage', 'pepper', 'rucola', 'walnut', 'mozzarella salad', 'oven vegetables', 'zuchetti', 'eggplant', 'piccata mass', 'bramata slice', 'vegetables for piccata', 'cream', 'chocolate cake', 'fried rice', 'beef meatballs', 'spicy vegetable ragout', 'olives', 'cucumber', 'tomato', 'turkey breast', 'carrot', 'chili with vegetables', 'lenses brown', 'lenses', 'pear', 'apple', 'apricot tart', 'cheese crêpe', 'meatloaf', 'peas', 'mashed potatoes', 'soggy bread without crust', 'big square plate', 'oversoaked polenta', 'soup of the day mushroom cream', 'cognac sauce', 'grated cheese', 'oversoaked chickpea curry', 'oversoaked roast beef', 'oversoaked food in pyramid shape', 'swiss chard vegetable ragout', 'oversoaked chia pudding', 'oversoaked mixed roast beef', 'cheese sauce', ' swiss chard', 'oversoaked mixed chickpea curry', 'sardinian fregola', 'smoked sausage (landjaeger)', 'soup of the day leek cream', 'vegetables for fregola', 'mustard', 'radish salad', 'fresh cheese praline', 'pickled cucumber', 'soup of the day banana-coconut', 'bean cassoulet', 'salami', 'pickled vegetables', 'deli meat cheese', 'turkey', 'sour cream', 'cylindrical transparent shot-glass', 'ricotta tortellini', 'potato vegetable curry', 'soup of the day sweetcorn', 'baked chickpea', 'tomato sauce', 'milk coffee', 'cherry jam', 'spreadable cheese', 'coffee cup', 'coffee plate', 'coffee yogurt', 'tilster cheese', 'brie cheese', 'margarine', 'appenzeller cheese', 'paprika sauce', 'bramata', 'green spelt dumplings', 'vegetable ragout for green spelt dumplings', 'chicken thigh steak', 'country cuts', 'veggie crispy bites', 'soup of the day tomatoes', 'vegetable salad for ham', 'country smoked ham', 'lye rolls', 'antipasti vegetables', 'tagliatelle tomato pesto antipasti', 'soup of the day beetroot', 'soy yogurt dip', 'vegan meatballs', 'vegetables for meatballs', 'yellow pea puree', 'boiled beef', 'horseradish bouillon', 'beef lasagna', 'eggplant cordon bleu', 'saffron risotto', 'bell pepper stew', 'pineapple-quark-mousse', 'quinoa salad', 'dried tomatoes', 'endives orange salad', 'orange fillet', 'cashew nuts', 'mascarpone', 'shiitake', 'red onion', 'risotto', 'vegetable salad', 'quinoa', 'little glass bowl', 'vegetable salad for quinoa', 'thai glass noodle salad', 'cheese ravioli', 'fruit quark', 'halibut', 'hummus', 'vegetables for halibut', 'asian dip', 'potato hash brown (roesti) with vegetables', 'oversoaked salmon fillet', 'oversoaked chickpea puree', 'bell pepper', 'port wine pears rucola risotto', 'soup of the day barley', 'gorgonzola', 'creamy polenta medium', 'beef patties in juicy sauce', 'merlot sauce', 'oversoaked perch fillet', 'oversoft food in crescent shape', 'rocket risotto', 'oversoaked bell peppers', 'oversoakeboiled beef', 'oversoakeboiled polenta', 'oversoaked carrots', 'polenta', 'soft zuchetti', 'potato herb patties', 'veggie cervalat sausage', 'colorful vegetables for veggie cervalat sausage', 'grisons barley soup', 'carrot puree', 'cod', 'ratatouille', 'soup of the day cauliflower cream', 'herbal semolina slice', 'crispy vegetable roll', 'rice noodle salad', 'soup of the day parmesan foam', 'creamed spinach', 'gnocchi pan tofu', 'homemade fishburgers', 'black bean puree', 'soup of the day zucchetti', 'roast beef', 'pizokel vegetable gratin', 'egg vinaigrette', 'cheesy soft egg noodles (kaesespaetzle)', 'cabbage salad', 'fennel salad for bowl', 'tree nut dressing', 'spelt marinated', 'feta marinated', 'beetroot cooked', 'iceberg lettuce', 'oversoaked smoked salmon', 'softened panna cotta', 'protein bowl', 'oversoaked fennel', 'oversoaked couscous', 'oversoaked turkey plate', 'lemon roulade', 'wild rice raw', 'homemade veggie burger', 'spicy tomato vegetable sauce', 'champignon organic', 'pork steak', 'gruyère', 'swiss macaroni and cheese', 'cantadou cheese', 'white bean puree', 'tilapia', 'burrito', 'pernod sauce', 'diced tomatoes', 'sliced beef', 'knot rolls', 'spelt goulash', 'french dressing', 'beans', 'salad', 'minced poultry patties', 'tofu', 'carbonara', 'carbonara tofu', 'rosemary sauce')


NUM_CLASSES = len(CLASSES)

evaluation = dict(
    interval=1,  # Evaluate after every epoch
    metric="bbox",  # Use bounding box metrics
    classwise=True,  # Enables per-class AP logging
)
auto_scale_lr = dict(base_batch_size=32, enable=True)
backend_args = None
data_root = "/opt/ml/input/data/"
dataset_type = "CocoDataset"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", interval=5, by_epoch=True, max_keep_ckpts=10
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook", draw=True, interval=10, show=False),
)
default_scope = "mmdet"
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)
lang_model_name = "bert-base-uncased"
launcher = "none"
load_from = LOAD_FROM
log_level = "INFO"
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)
max_epochs = MAX_EPOCHS
metainfo = dict(classes=CLASSES)

# #ToDo: remove....?
# class_weight = [1.0, 1.0, 0.89, 1.0, 0.14, 1.0, 1.0, 0.63, 0.11, 1.0, 0.48, 0.25, 0.19, 0.29, 0.53, 1.0, 1.0, 1.0, 0.33, 1.0, 0.09, 1.0, 1, 0.09, 1.0, 0.5, 0.38, 0.54, 1.0, 1.0, 1.0, 0.9, 1.0, 0.97, 0.98, 0.96, 0.3, 1.0, 1.0, 0.8, 0.95, 0.17, 1.0, 0.8, 1.0, 1.0, 0.83, 1.0, 1.0, 1.0, 1.0, 1.0, 0.57, 1.0, 1.0, 0.69, 1.0, 0.83, 0.45, 1.0, 1.0, 1.0, 1.0, 1.0, 0.17, 1.0, 1.0, 0.54, 1.0, 0.46, 0.6, 1.0, 1.0, 0.62, 0.83, 0.68, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.33, 1.0, 0.75, 0.32, 0.24, 0.75, 1.0, 1.0, 0.67, 0.56, 0.77, 0.17, 0.85, 0.31, 0.8, 1.0, 0.67, 1.0, 1.0, 1.0, 0.69, 1.0, 0.9, 0.59, 0.8, 0.15, 1.0, 1.0, 0.75, 0.12, 0.8, 0.23, 0.86, 0.75, 0.57, 0.8, 0.8, 1.0, 1.0, 1.0, 0.67, 0.46, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.67, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 0.39, 1.0, 1.0, 1.0, 1.0, 0.28, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.67, 1.0, 1.0, 0.17, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.41, 0.29, 1.0, 0.15, 0.83, 1.0, 1.0, 1.0, 1.0, 0.42, 1.0, 0.19, 1.0, 0.16, 1.0, 1.0, 0.42, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.62, 1.0, 0.23, 1.0, 1.0, 0.55, 0.8, 0.7, 0.86, 0.63, 1.0, 1.0, 0.82, 1.0, 1.0, 1.0, 1.0, 0.77, 0.14, 1.0, 0.12, 0.21, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 0.33, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 0.5, 1.0, 0.39, 0.45, 0.37, 0.58, 0.28, 1.0, 0.83, 1.0, 1.0, 1.0, 0.92, 1.0, 0.6, 1.0, 0.82, 0.45, 1.0, 0.67, 1.0, 1.0, 1.0, 1.0, 0.6, 0.67, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.36, 1.0, 1.0, 1.0, 0.2, 0.5, 0.71, 1.0, 1.0, 1.0, 0.15, 0.67, 0.42, 0.86, 0.89, 0.16, 1.0, 1.0, 0.67, 1.0, 1.0, 0.92, 1.0, 1.0, 1.0, 1.0, 0.42, 1.0, 1.0, 0.33, 1.0, 0.83, 0.3, 1.0, 1.0, 0.1, 0.33, 1.0, 1.0, 1.0, 0.43, 0.18, 0.5, 1.0, 0.75, 0.33, 1.0, 1.0, 0.62, 1.0, 1.0, 1.0, 1.0, 1.0, 0.87, 1.0, 1.0, 1.0, 0.38, 0.69, 1.0, 1.0, 1.0, 1.0, 0.67, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.12, 1.0, 0.13, 1.0, 1.0, 0.31, 1.0, 0.75, 0.38, 1.0, 1.0, 0.75, 0.44, 0.75, 0.24, 0.44, 1.0, 1.0, 0.08, 1.0, 1.0, 1.0, 1.0, 0.36, 0.45, 1.0, 0.67, 1.0, 0.6, 1.0, 1.0, 0.67, 0.4, 1.0, 1.0, 1.0, 1.0, 0.5, 0.33, 1.0, 0.29, 0.38, 1.0, 0.8, 1.0, 0.67, 1.0, 0.67, 0.33, 0.25, 1.0, 0.22, 1.0, 1.0, 1.0, 1.0, 1.0, 0.54, 1.0, 0.5, 1.0, 0.12, 1.0, 1.0, 1.0, 0.5, 0.75, 0.5, 0.6, 1.0, 1.0, 1.0, 1.0, 0.25, 0.8, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]

model = dict(
    all_labels=CLASSES,
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=False,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type="SwinTransformer",
        window_size=7,
        with_cp=True,
    ),
    bbox_head=dict(
        contrastive_cfg=dict(bias=False, log_scale=0.0, max_text_len=256),
        loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
        loss_cls=dict(
            alpha=0.25, gamma=2.0, loss_weight=1.0, type="FocalLoss", use_sigmoid=True
        ),
        loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
        # ToDo: set the number of classes automatically.
        num_classes=NUM_CLASSES,
        sync_cls_avg_factor=True,
        type="GroundingDINOHead",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=False,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type="DetDataPreprocessor",
    ),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            cross_attn_text_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
        ),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True,
    ),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5,
    ),
    encoder=dict(
        fusion_layer_cfg=dict(
            embed_dim=1024, init_values=0.0001, l_dim=256, num_heads=4, v_dim=256
        ),
        layer_cfg=dict(
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
        ),
        num_cp=6,
        num_layers=6,
        text_layer_cfg=dict(
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=4),
        ),
    ),
    language_model=dict(
        add_pooling_layer=False,
        name="bert-base-uncased",
        pad_to_max=False,
        special_tokens_list=[
            "[CLS]",
            "[SEP]",
            ".",
            "?",
        ],
        type="BertModel",
        use_sub_sentence_represent=True,
    ),
    neck=dict(
        act_cfg=None,
        bias=True,
        in_channels=[
            192,
            384,
            768,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type="GN"),
        num_outs=4,
        out_channels=256,
        type="ChannelMapper",
    ),
    num_queries=900,
    positional_encoding=dict(normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type="BinaryFocalLossCost", weight=2.0),
                dict(box_format="xywh", type="BBoxL1Cost", weight=5.0),
                dict(iou_mode="giou", type="IoUCost", weight=2.0),
            ],
            type="HungarianAssigner",
        )
    ),
    type="GroundingDINO",
    with_box_refine=True,
)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type="AdamW", weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0), backbone=dict(lr_mult=0.1)
        )
    ),
    type="OptimWrapper",
)

param_scheduler = [
    # Warm-up scheduler
    dict(
        type="LinearLR",  # Linear warm-up
        start_factor=0.001,  # Starting LR is 0.1% of the base LR
        by_epoch=False,  # Apply warm-up by iteration, not by epoch
        begin=0,  # Start from the very first iteration
        end=50,  # End at the 50th iteration # original: 250
    )
    # Uncomment the section below if you want to apply a linear decay after warm-up
    # dict(
    #     type='LinearLR',         # Linear learning rate decay
    #     start_factor=1.0,        # Start at full base LR after warm-up
    #     end_factor=0.01,         # Decay to 1% of the base LR
    #     by_epoch=True,           # Apply decay by epoch
    #     begin=51,                # Start decay right after warm-up ends
    #     end=36                   # End decay at the 36th epoch
    # )
]

resume = RESUME

train_cfg = dict(max_epochs=max_epochs, type="EpochBasedTrainLoop", val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=BATCH_SIZE_TRAIN,
    dataset=dict(
        type="CocoDataset",
        metainfo=metainfo,
        ann_file=ANN_FILE_TRAINING,
        backend_args=None,
        data_prefix=DATA_PREFIX_TRAIN,
        data_root="/opt/ml/input/data/",
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="LoadTextAnnotations", classes=CLASSES),
            dict(prob=0.5, type="RandomFlip"),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type="RandomChoiceResize",
                        ),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    400,
                                    4200,
                                ),
                                (
                                    500,
                                    4200,
                                ),
                                (
                                    600,
                                    4200,
                                ),
                            ],
                            type="RandomChoiceResize",
                        ),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type="absolute_range",
                            type="RandomCrop",
                        ),
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type="RandomChoiceResize",
                        ),
                    ],
                ],
                type="RandomChoice",
            ),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                    "flip",
                    "flip_direction",
                    "text",
                    "custom_entities",
                ),
                type="PackDetInputs",
            ),
        ],
        return_classes=True,
    ),
    num_workers=NUM_WORKER_TRAIN,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)

val_dataloader = dict(
    batch_size=BATCH_SIZE_VAL,
    num_workers=NUM_WORKER_VAL,
    persistent_workers=True,
    dataset=dict(
        type="CocoDataset",
        metainfo=metainfo,
        # ToDo: load the validation set name dynamically
        ann_file=ANN_FILE_VALIDATION,  # Validation annotations
        data_prefix=DATA_PREFIX_VAL,  # Validation images
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    800,
                    1333,
                ),
                type="FixScaleResize",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="LoadTextAnnotations", classes=CLASSES),  # For GroundingDINO
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                    "flip",
                    "flip_direction",
                    "text",
                    "custom_entities",
                ),
                type="PackDetInputs",
            ),
        ],
    ),
    sampler=dict(shuffle=False, type="DefaultSampler"),  # No shuffling for validation
)

val_cfg = dict(type="ValLoop")

val_evaluator = dict(
    type="OpenSetCOCOMetric",
    ann_file=ANN_FILE_VALIDATION,
    classes=CLASSES,
    metric=["bbox"],  # Metrics for both bounding boxes and segmentation
    classwise=True,  # Enable class-wise mAP
)

vis_backends = [
    dict(type="LocalVisBackend", save_dir="/opt/ml/output/data/visualizations"),
]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
    save_dir="/opt/ml/output/data/visualizations",
)
work_dir = "/opt/ml/checkpoints"

test_dataloader = dict(
    batch_size=BATCH_SIZE_VAL,
    num_workers=NUM_WORKER_VAL,
    persistent_workers=True,
    dataset=dict(
        type="CocoDataset",
        metainfo=metainfo,
        ann_file=ANN_FILE_VALIDATION,  # Using validation set for testing
        data_prefix=DATA_PREFIX_VAL,  # Test images from validation dataset
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    800,
                    1333,
                ),
                type="FixScaleResize",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="LoadTextAnnotations", classes=CLASSES),  # For GroundingDINO
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                    "flip",
                    "flip_direction",
                    "text",
                    "custom_entities",
                ),
                type="PackDetInputs",
            ),
        ],
    ),
    sampler=dict(shuffle=False, type="DefaultSampler"),  # No shuffling for test
)

test_cfg = dict(type="TestLoop")

custom_imports = dict(
    imports=[
        "mmdet.evaluation.metrics.coco_metric_open_set_detection"
    ],  # Full module path
    allow_failed_imports=False,  # Ensures import failure raises an error
)

test_evaluator = dict(
    type="OpenSetCOCOMetric",
    ann_file=ANN_FILE_VALIDATION,
    classes=CLASSES,
    metric=["bbox"],  # Metrics for bounding boxes
    classwise=True,  # Enable class-wise mAP for detailed evaluation
)
