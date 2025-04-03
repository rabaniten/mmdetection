import os
import json
import hashlib
import warnings
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.registry import METRICS
import torch

#ToDo: not not hardcode label here, check how to access them.
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

@METRICS.register_module()
class OpenSetCOCOMetric(CocoMetric):
    """Custom COCO Evaluator for Open-Set Detection Models in MMDetection."""

    def __init__(self, ann_file, outfile_prefix=None, **kwargs):
        super().__init__(ann_file=ann_file, outfile_prefix=outfile_prefix, **kwargs)

        self.ann_file = ann_file 
        print(f"🔍 OpenSetCOCOMetric initialized with annotation file: {self.ann_file}", flush=True)
        
        # Load COCO ground truth annotations
        self.coco_gt = COCO(self.ann_file)

        # Mapping: category name → internal label index
        self.global_prompt_to_index = {
            name: idx for idx, name in enumerate(ALL_LABELS)
        }

        print(f"✅ Loaded {len(self.global_prompt_to_index)} categories from COCO annotations.", flush=True)

    def process(self, data_batch, data_samples):
        """Convert predictions into COCO format before calling standard COCO processing."""
        updated_data_samples = []

        for data_sample in data_samples:
            img_id = data_sample.get("img_id", None)

            pred_instances = data_sample.get("pred_instances", None)
            if pred_instances is None:
                print(f"⚠️  No pred_instances found for image ID {img_id}, skipping.", flush=True)
                continue

            if "labels" not in pred_instances:
                print(f"⚠️  No 'labels' in pred_instances for image ID {img_id}, skipping.", flush=True)
                continue

            text_prompt = data_sample.get("text", None)
            if text_prompt is None:
                print(f"⚠️  No 'text' found for image ID {img_id}, skipping.", flush=True)
                continue

            mapped_labels = []
            for label_idx in pred_instances["labels"].tolist():
                try:
                    category_name = text_prompt[label_idx]
                    category_id = self.global_prompt_to_index.get(category_name, -1)

                    if category_id == -1:
                        warnings.warn(f"⚠️ Category '{category_name}' not found in COCO categories.")

                    mapped_labels.append(category_id)
          
                except Exception as e:
                    print(f"❌ Error processing label index {label_idx}: {e}", flush=True)
                    continue

            pred_instances["labels"] = torch.tensor(mapped_labels, dtype=torch.int64, device='cuda')
            data_sample["pred_instances"] = pred_instances
            updated_data_samples.append(data_sample)

        print(f"📦 Passing {len(updated_data_samples)} updated samples to parent CocoMetric.", flush=True)
        super().process(data_batch, updated_data_samples)
