# Local training and inference
LOAD_FROM = '/root/Sofia/Genioos/sofia_thesis_project/detection_models/grounding_dino/pretrained_models/custom_grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544-5f234b20.pth'

RESUME = False

ANN_FILE_TRAINING = '/root/Sofia/Genioos/data/Stadtspital-Waid/annotated_data_for_ml_model/training_and_val_data/coco_training/coco_train_annotations.json'
ANN_FILE_VALIDATION = '/root/Sofia/Genioos/data/Stadtspital-Waid/annotated_data_for_ml_model/training_and_val_data/coco_validation/coco_val_annotations.json'

DATA_PREFIX_TRAIN = dict(img= '/root/Sofia/Genioos/data/Stadtspital-Waid/annotated_data_for_ml_model/training_and_val_data/coco_training/images/')
DATA_PREFIX_VAL = dict(img='/root/Sofia/Genioos/data/Stadtspital-Waid/annotated_data_for_ml_model/training_and_val_data/coco_validation/images/')

BATCH_SIZE_TRAIN = 1
BATCH_SIZE_VAL = 1

NUM_WORKER_TRAIN = 2
NUM_WORKER_VAL = 2



# # Training and inference in custom docker
# LOAD_FROM = '/opt/ml/code/pretrained_models/groundingdino_swint_ogc_mmdet-822d7e9d.pth'

# RESUME = False

# ANN_FILE_TRAINING = '/opt/ml/input/data/train/instance_seg_train_no_crowd.json'
# ANN_FILE_VALIDATION = '/opt/ml/input/data/validation/instance_seg_val_no_crowd.json'

# DATA_PREFIX_TRAIN = dict(img= '/opt/ml/input/data/train/images/')
# DATA_PREFIX_VAL = dict(img='/opt/ml/input/data/validation/images/')

# BATCH_SIZE_TRAIN = 1
# BATCH_SIZE_VAL = 1

# NUM_WORKER_TRAIN = 2
# NUM_WORKER_VAL = 2



CLASSES = ('chervil cream sauce', 'small sauce-glass', 'cream tart', 'white plate', 'chicken breast', 'tortellini', 'shallow bowl', 'cherry tomato', 'caesar salad', 'parmesan dressing', 'egg cooked', 'toast croutons', 'parmesan shavings', 'bacon', 'lettuce', 'dressing', 'colorful vegetable pan with soft egg noodles (spaetzle)', 'white plate without rim', 'capuns', 'soup of the day ratatouille cream', 'root vegetables', 'soup-bowl', 'beer sauce', 'finger-shaped potato dumplings (schupfnudeln)', 'sauerkraut', 'smoked pork neck', 'roasted cashew nuts', 'hioumi', 'raspberry quark', 'natural yogurt', 'strawberry yogurt', 'apricot yogurt', 'raspberry yogurt', 'beans green', 'cream sauce', 'vegetable bolognese', 'potatoes', 'thyme', 'horn-shaped pasta (hoernli)', 'caramel flan', 'big sauce-glass', 'fruit salad', 'soup of the day artichoke', 'veal sausage', 'onion sauce', 'glass fruitsalad-bowl', 'spanish tortilla', 'basil sauce', 'vanilla cream puffs', 'small quadratic plate-bowl', 'basil pesto', 'quadratic dessert-plate', 'radish', 'sausage and cheese salad', 'lollo bianco', 'house bread', 'soup of the day potato', 'vegetarian bami goreng', 'lollo rosso', 'soup of the day broccoli cream', 'spaghetti', 'vegetable strips', 'saffron herb sauce', 'vegetable strips saffron-herb sauce', 'pureed green balls', 'pureed food in a special shape', 'eggplant moussaka', 'pureed meat slices', 'pureed food in oval shape', 'pureed broccoli', 'pureed mashed potatoes', 'spinach tart', 'turmeric', 'pita bread', 'trout fillet', 'spinach', 'rice', 'sauce', 'sprout vegetables', 'scallion', 'herbal rice', 'soup of the day curry cream', 'lentil ragout', 'paneer', 'quinoa patties', 'vegetables for quinoa patties', 'toast', 'turkey ham', 'pineapple', 'gruyere cheese', 'onion red', 'cranberry', 'barley risotto', 'salad leaves', 'cheese tart', 'zucchetti', 'lemon panna cotta', 'parsley fritters', 'lemon', 'capers', 'smoked trout', 'trout tartare', 'soup of the day yellow pea', 'horseradish foam', 'butter', 'vegetable lasagna', 'applesauce', 'broccoli', 'dill mashed potatoes', 'salmon cubes marinated', 'white wine sauce', 'brown sauce', 'pureed food in pyramid shape', 'penne rigate', 'pureed polenta', 'pureed balls', 'tagliatelle', 'pureed cauliflower', 'raspberry', 'raspberry mousse in pyramid shape', 'bolognese', 'white sauce', 'bell pepper sauce', 'chocolate mousse', 'round raspberry mousse', 'romanesco', 'slices', 'poultry stew', 'pureed salmon', 'pureed chicken thigh', 'pureed fries', 'pureed sausage', 'penne', 'plate with red rim', 'soft egg noodles (spaetzle)', 'veggie swiss macaroni and cheese', 'poulet', 'pasta', 'boiled meat salad seed oil', 'vegetable strudel', 'vegetable patch', 'herb quark dip', 'vegetables for boiled meat salad', 'plum tart', 'boiled meat', 'gnocchi seitan pan', 'lamb stew', 'bulgur', 'carrots', 'currant sheet cake', 'bulgur sauce', 'gnocchi', 'sliced seitan', 'oyster mushrooms', 'vegetable salad with white beans', 'vegetables for green spelt risotto', 'green spelt risotto', 'rye bread', 'apple tart', 'bouillon', 'cold chicken breast', 'cocktail sauce', 'soup of the day carrot cream', 'curry dip', 'soup of the day lentil ginger', 'poulet cordon bleu', 'jus', 'pilau rice', 'sugar peas', 'roasted cauliflower', 'sauce for sliced seitan', 'boiled potatoes', 'semolina porridge', 'cherry compote', 'cinnamon sugar', 'small plastic cup', 'oversoaked sliced chicken', 'overly soft thick brie cheese', 'overly soft cottage cheese', 'meat cheese', 'mustard sauce', 'lyonnaise potatoes', 'oversoaked sliced veal', 'overly soft cream cheese', 'overly soft thin brie cheese', 'currants', 'soup of the day bell peppers', 'sliced quorn sauce zurich style', 'sliced quorn zurich style', 'colorful vegetables from zuchetti peas carrots and beans', 'hash brown (roesti)', 'bread dumplings', 'sauce poultry ragout', 'sliced quorn', 'vegetables', 'banana organic', 'croissant', 'lye croissant', 'coffee', 'lid on the ground', 'uncovered jug', 'jug covered with lid', 'orange juice', 'multigrain roll', 'scrambled eggs', 'milk', 'mueesli', 'large glass fruitsalad-bowl', 'milk roll', 'mozzarella', 'baked vegetables for mozzarella', 'wedges', 'chipolata sausage', 'vegetables for chipolata sausage', 'pepper', 'rucola', 'walnut', 'mozzarella salad', 'oven vegetables', 'zuchetti', 'eggplant', 'piccata mass', 'bramata slice', 'vegetables for piccata', 'cream', 'chocolate cake', 'fried rice', 'beef meatballs', 'spicy vegetable ragout', 'olives', 'cucumber', 'tomato', 'turkey breast', 'carrot', 'chili with vegetables', 'lenses brown', 'lenses', 'pear', 'apple', 'apricot tart', 'cheese crêpe', 'meatloaf', 'peas', 'mashed potatoes', 'soggy bread without crust', 'big square plate', 'oversoaked polenta', 'soup of the day mushroom cream', 'cognac sauce', 'grated cheese', 'oversoaked chickpea curry', 'oversoaked roast beef', 'oversoaked food in pyramid shape', 'swiss chard vegetable ragout', 'oversoaked chia pudding', 'oversoaked mixed roast beef', 'cheese sauce', ' swiss chard', 'oversoaked mixed chickpea curry', 'sardinian fregola', 'smoked sausage (landjaeger)', 'soup of the day leek cream', 'vegetables for fregola', 'mustard', 'radish salad', 'fresh cheese praline', 'pickled cucumber', 'soup of the day banana-coconut', 'bean cassoulet', 'salami', 'pickled vegetables', 'deli meat cheese', 'turkey', 'sour cream', 'cylindrical transparent shot-glass', 'ricotta tortellini', 'potato vegetable curry', 'soup of the day sweetcorn', 'baked chickpea', 'tomato sauce', 'milk coffee', 'cherry jam', 'spreadable cheese', 'coffee cup', 'coffee plate', 'coffee yogurt', 'tilster cheese', 'brie cheese', 'margarine', 'appenzeller cheese', 'paprika sauce', 'bramata', 'green spelt dumplings', 'vegetable ragout for green spelt dumplings', 'chicken thigh steak', 'country cuts', 'veggie crispy bites', 'soup of the day tomatoes', 'vegetable salad for ham', 'country smoked ham', 'lye rolls', 'antipasti vegetables', 'tagliatelle tomato pesto antipasti', 'soup of the day beetroot', 'soy yogurt dip', 'vegan meatballs', 'vegetables for meatballs', 'yellow pea puree', 'boiled beef', 'horseradish bouillon', 'beef lasagna', 'eggplant cordon bleu', 'saffron risotto', 'bell pepper stew', 'pineapple-quark-mousse', 'quinoa salad', 'dried tomatoes', 'endives orange salad', 'orange fillet', 'cashew nuts', 'mascarpone', 'shiitake', 'red onion', 'risotto', 'vegetable salad', 'quinoa', 'little glass bowl', 'vegetable salad for quinoa', 'thai glass noodle salad', 'cheese ravioli', 'fruit quark', 'halibut', 'hummus', 'vegetables for halibut', 'asian dip', 'potato hash brown (roesti) with vegetables', 'oversoaked salmon fillet', 'oversoaked chickpea puree', 'bell pepper', 'port wine pears rucola risotto', 'soup of the day barley', 'gorgonzola', 'creamy polenta medium', 'beef patties in juicy sauce', 'merlot sauce', 'oversoaked perch fillet', 'oversoft food in crescent shape', 'rocket risotto', 'oversoaked bell peppers', 'oversoakeboiled beef', 'oversoakeboiled polenta', 'oversoaked carrots', 'polenta', 'soft zuchetti', 'potato herb patties', 'veggie cervalat sausage', 'colorful vegetables for veggie cervalat sausage', 'grisons barley soup', 'carrot puree', 'cod', 'ratatouille', 'soup of the day cauliflower cream', 'herbal semolina slice', 'crispy vegetable roll', 'rice noodle salad', 'soup of the day parmesan foam', 'creamed spinach', 'gnocchi pan tofu', 'homemade fishburgers', 'black bean puree', 'soup of the day zucchetti', 'roast beef', 'pizokel vegetable gratin', 'egg vinaigrette', 'cheesy soft egg noodles (kaesespaetzle)', 'cabbage salad', 'fennel salad for bowl', 'tree nut dressing', 'spelt marinated', 'feta marinated', 'beetroot cooked', 'iceberg lettuce', 'oversoaked smoked salmon', 'softened panna cotta', 'protein bowl', 'oversoaked fennel', 'oversoaked couscous', 'oversoaked turkey plate', 'lemon roulade', 'wild rice raw', 'homemade veggie burger', 'spicy tomato vegetable sauce', 'champignon organic', 'pork steak', 'gruyère', 'swiss macaroni and cheese', 'cantadou cheese', 'white bean puree', 'tilapia', 'burrito', 'pernod sauce', 'diced tomatoes', 'sliced beef', 'knot rolls', 'spelt goulash', 'french dressing', 'beans', 'salad', 'minced poultry patties', 'tofu', 'carbonara', 'carbonara tofu', 'rosemary sauce')


NUM_CLASSES = len(CLASSES)

evaluation = dict(
    interval=1,         # Evaluate after every epoch
    metric='bbox',       # Use bounding box metrics
    classwise=True       # Enables per-class AP logging
)
auto_scale_lr = dict(base_batch_size=32, enable=True)
backend_args = None
data_root = '/opt/ml/input/data/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook', by_epoch=True, max_keep_ckpts=3),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
lang_model_name = 'bert-base-uncased'
launcher = 'none'
load_from = LOAD_FROM
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 40
metainfo = dict(
    classes= CLASSES
)

#ToDo: remove....?
class_weight = [1.0, 1.0, 0.89, 1.0, 0.14, 1.0, 1.0, 0.63, 0.11, 1.0, 0.48, 0.25, 0.19, 0.29, 0.53, 1.0, 1.0, 1.0, 0.33, 1.0, 0.09, 1.0, 1, 0.09, 1.0, 0.5, 0.38, 0.54, 1.0, 1.0, 1.0, 0.9, 1.0, 0.97, 0.98, 0.96, 0.3, 1.0, 1.0, 0.8, 0.95, 0.17, 1.0, 0.8, 1.0, 1.0, 0.83, 1.0, 1.0, 1.0, 1.0, 1.0, 0.57, 1.0, 1.0, 0.69, 1.0, 0.83, 0.45, 1.0, 1.0, 1.0, 1.0, 1.0, 0.17, 1.0, 1.0, 0.54, 1.0, 0.46, 0.6, 1.0, 1.0, 0.62, 0.83, 0.68, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.33, 1.0, 0.75, 0.32, 0.24, 0.75, 1.0, 1.0, 0.67, 0.56, 0.77, 0.17, 0.85, 0.31, 0.8, 1.0, 0.67, 1.0, 1.0, 1.0, 0.69, 1.0, 0.9, 0.59, 0.8, 0.15, 1.0, 1.0, 0.75, 0.12, 0.8, 0.23, 0.86, 0.75, 0.57, 0.8, 0.8, 1.0, 1.0, 1.0, 0.67, 0.46, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.67, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 0.39, 1.0, 1.0, 1.0, 1.0, 0.28, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.67, 1.0, 1.0, 0.17, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.41, 0.29, 1.0, 0.15, 0.83, 1.0, 1.0, 1.0, 1.0, 0.42, 1.0, 0.19, 1.0, 0.16, 1.0, 1.0, 0.42, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.62, 1.0, 0.23, 1.0, 1.0, 0.55, 0.8, 0.7, 0.86, 0.63, 1.0, 1.0, 0.82, 1.0, 1.0, 1.0, 1.0, 0.77, 0.14, 1.0, 0.12, 0.21, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 0.33, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 0.5, 1.0, 0.39, 0.45, 0.37, 0.58, 0.28, 1.0, 0.83, 1.0, 1.0, 1.0, 0.92, 1.0, 0.6, 1.0, 0.82, 0.45, 1.0, 0.67, 1.0, 1.0, 1.0, 1.0, 0.6, 0.67, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.36, 1.0, 1.0, 1.0, 0.2, 0.5, 0.71, 1.0, 1.0, 1.0, 0.15, 0.67, 0.42, 0.86, 0.89, 0.16, 1.0, 1.0, 0.67, 1.0, 1.0, 0.92, 1.0, 1.0, 1.0, 1.0, 0.42, 1.0, 1.0, 0.33, 1.0, 0.83, 0.3, 1.0, 1.0, 0.1, 0.33, 1.0, 1.0, 1.0, 0.43, 0.18, 0.5, 1.0, 0.75, 0.33, 1.0, 1.0, 0.62, 1.0, 1.0, 1.0, 1.0, 1.0, 0.87, 1.0, 1.0, 1.0, 0.38, 0.69, 1.0, 1.0, 1.0, 1.0, 0.67, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.12, 1.0, 0.13, 1.0, 1.0, 0.31, 1.0, 0.75, 0.38, 1.0, 1.0, 0.75, 0.44, 0.75, 0.24, 0.44, 1.0, 1.0, 0.08, 1.0, 1.0, 1.0, 1.0, 0.36, 0.45, 1.0, 0.67, 1.0, 0.6, 1.0, 1.0, 0.67, 0.4, 1.0, 1.0, 1.0, 1.0, 0.5, 0.33, 1.0, 0.29, 0.38, 1.0, 0.8, 1.0, 0.67, 1.0, 0.67, 0.33, 0.25, 1.0, 0.22, 1.0, 1.0, 1.0, 1.0, 1.0, 0.54, 1.0, 0.5, 1.0, 0.12, 1.0, 1.0, 1.0, 0.5, 0.75, 0.5, 0.6, 1.0, 1.0, 1.0, 1.0, 0.25, 0.8, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]

model = dict(
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
        type='SwinTransformer',
        window_size=7,
        with_cp=True),
    bbox_head=dict(
        contrastive_cfg=dict(bias=False, log_scale=0.0, max_text_len=256),
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        # ToDo: set the number of classes automatically.
        num_classes=NUM_CLASSES,
        sync_cls_avg_factor=True,
        type='GroundingDINOHead'),
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
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            cross_attn_text_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        fusion_layer_cfg=dict(
            embed_dim=1024,
            init_values=0.0001,
            l_dim=256,
            num_heads=4,
            v_dim=256),
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_cp=6,
        num_layers=6,
        text_layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=4))),
    language_model=dict(
        add_pooling_layer=False,
        name='bert-base-uncased',
        pad_to_max=False,
        special_tokens_list=[
            '[CLS]',
            '[SEP]',
            '.',
            '?',
        ],
        type='BertModel',
        use_sub_sentence_represent=True),
    neck=dict(
        act_cfg=None,
        bias=True,
        in_channels=[
            192,
            384,
            768,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='GroundingDINO',
    with_box_refine=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')

param_scheduler = [
    # Warm-up scheduler
    dict(
        type='LinearLR',          # Linear warm-up
        start_factor=0.001,       # Starting LR is 0.1% of the base LR
        by_epoch=False,           # Apply warm-up by iteration, not by epoch
        begin=0,                  # Start from the very first iteration
        end=50                    # End at the 50th iteration # original: 250
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

train_cfg = dict(max_epochs = max_epochs, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=BATCH_SIZE_TRAIN,
    dataset=dict(
        #ToDo: Set the filename from the jupyter notebook (instead of hardcoing here)
        #Define name of annotation file
        ann_file=ANN_FILE_TRAINING,
        backend_args=None,
        data_prefix=DATA_PREFIX_TRAIN,
        data_root='/opt/ml/input/data/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='LoadTextAnnotations'),  # new
            dict(prob=0.5, type='RandomFlip'),
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
                            type='RandomChoiceResize'),
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
                            type='RandomChoiceResize'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type='absolute_range',
                            type='RandomCrop'),
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
                            type='RandomChoiceResize'),
                    ],
                ],
                type='RandomChoice'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'text',
                    'custom_entities',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        type='CocoDataset'),
    num_workers=NUM_WORKER_TRAIN,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=BATCH_SIZE_VAL,
    num_workers=NUM_WORKER_VAL,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        # ToDo: load the validation set name dynamically
        ann_file=ANN_FILE_VALIDATION,  # Validation annotations
        data_prefix=DATA_PREFIX_VAL,  # Validation images
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='LoadTextAnnotations'),  # For GroundingDINO
             dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'text',
                    'custom_entities',
                ),
                type='PackDetInputs'),
        ]
    ),
    sampler=dict(shuffle=False, type='DefaultSampler')  # No shuffling for validation
)

val_cfg = dict(type='ValLoop')  

val_evaluator = dict(
    type='CocoMetric',
    ann_file=ANN_FILE_VALIDATION,
    metric=['bbox'],  # Metrics for both bounding boxes and segmentation
    classwise=True            # Enable class-wise mAP
)

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/opt/ml/checkpoints'

test_dataloader = dict(
    batch_size=BATCH_SIZE_VAL,
    num_workers=NUM_WORKER_VAL,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        ann_file=ANN_FILE_VALIDATION,  # Using validation set for testing
        data_prefix=DATA_PREFIX_VAL,  # Test images from validation dataset
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                1333,
            ), type='FixScaleResize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='LoadTextAnnotations'),  # For GroundingDINO
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'text',
                    'custom_entities',
                ),
                type='PackDetInputs'),
        ]
    ),
    sampler=dict(shuffle=False, type='DefaultSampler')  # No shuffling for test
)

test_cfg = dict(type='TestLoop')  

custom_imports = dict(
    imports=['mmdet.evaluation.metrics.coco_metric_open_set_detection'],  # Full module path
    allow_failed_imports=False  # Ensures import failure raises an error
)

test_evaluator = dict(
    type='OpenSetCOCOMetric',
    ann_file=ANN_FILE_VALIDATION,
    metric=['bbox'],  # Metrics for bounding boxes
    classwise=True  # Enable class-wise mAP for detailed evaluation
)
