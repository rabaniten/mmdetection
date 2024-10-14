auto_scale_lr = dict(base_batch_size=32, enable=True)
backend_args = None
data_root = '/opt/ml/input/data/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook', by_epoch=True, max_keep_ckpts=1),
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
load_from = '/opt/ml/code/pretrained_models/custom_grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544-5f234b20.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 24
metainfo = dict(
    classes=('chervil cream sauce', 'small sauce-glass', 'cream tart', 'white plate', 'chicken breast', 'tortellini', 'shallow bowl', 'cherry tomato', 'mixed dish of tortellini and chervil cream sauce', 'caesar salad', 'parmesan dressing', 'egg cooked', 'toast croutons', 'parmesan shavings', 'bacon', 'lettuce', 'mixed dish of caesar salad and dressing', 'colorful vegetable pan with soft egg noodles (spaetzle)', 'white plate without rim', 'capuns', 'soup of the day ratatouille cream', 'root vegetables', 'soup bowl', 'beer sauce', 'finger-shaped potato dumplings (schupfnudeln)', 'sauerkraut', 'smoked pork neck', 'roasted cashew nuts', 'hioumi', 'raspberry quark', 'natural yogurt', 'strawberry yogurt', 'apricot yogurt', 'raspberry yogurt', 'beans green', 'cream sauce', 'vegetable bolognese', 'potatoes', 'mixed dish of horn-shaped pasta (hoernli) and vegetable bolognese', 'caramel flan', 'large sauce-glass', 'fruit salad', 'soup of the day artichoke', 'veal sausage', 'onion sauce', 'small glass fruitsalad bowl', 'spanish tortilla with basil sauce', 'vanilla cream puffs', 'small quadratic plate-bowl', 'spanish tortilla', 'quadratic dessert-plate', 'horn-shaped pasta (hoernli)', 'radish', 'sausage and cheese salad', 'lollo bianco', 'house bread', 'soup of the day potato', 'vegetarian bami goreng', 'lollo rosso', 'soup of the day broccoli cream', 'mixed dish of spaghetti vegetable strips and saffron herb sauce', 'vegetable strips saffron-herb sauce', 'pureed green balls', 'spaghetti', 'pureed food in a special shape', 'eggplant moussaka', 'pureed meat slices', 'pureed food in oval shape', 'pureed broccoli', 'pureed mashed potatoes', 'spinach tart', 'turmeric sauce', 'pita bread', 'mixed dish of trout fillet spinach rice and sauce', 'sprout vegetables', 'scallion', 'herbal rice', 'soup of the day curry cream', 'lentil ragout', 'spinach', 'trout fillet', 'mixed meal of lentil ragout paneer pita and sprouts', 'quinoa patties', 'vegetables for quinoa patties', 'toast', 'filling of toast hawaii', 'onion red', 'cranberry', 'barley risotto', 'salad leaves', 'cheese tart', 'zucchetti', 'lemon panna cotta', 'parsley fritters', 'lemon', 'capers', 'smoked trout', 'trout tartare', 'soup of the day yellow pea', 'horseradish foam', 'butter', 'pesto sauce', 'vegetable lasagna', 'applesauce', 'broccoli', 'dill mashed potatoes', 'salmon cubes marinated', 'white wine sauce', 'brown sauce', 'pureed food in pyramid shape', 'penne rigate', 'pureed polenta', 'pureed balls', 'tagliatelle', 'pureed cauliflower', 'raspberry', 'raspberry mousse in pyramid shape', 'bolognese', 'white sauce', 'bell pepper sauce', 'chocolate mousse', 'round raspberry mousse', 'romanesco', 'slices', 'poultry stew', 'pureed salmon', 'pureed chicken thigh', 'pureed fries', 'pureed sausage', 'mixed dish of penne and bolognese', 'plate with red rim', 'mixed meal of soft egg noodles (spaetzle) and halloumi', 'veggie swiss macaroni and cheese', 'mixed dish of chicken bell pepper sauce romanesco and pasta', 'boiled meat salad seed oil', 'vegetable strudel', 'vegetable patch', 'herb quark dip', 'vegetables for boiled meat salad', 'plum tart', 'boiled meat', 'mixed meal of vegetable strudel and herb quark dip', 'gnocchi seitan pan', 'mixed meal of lamb stew bulgur and carrots', 'carrots', 'currant sheet cake', 'lamb stew', 'sauce', 'bulgur', 'mixed dish of gnocchi seitan pan and oyster mushrooms', 'vegetable salad with white beans', 'vegetables for green spelt risotto', 'green spelt risotto', 'rye bread', 'apple tart', 'bouillon', 'cold chicken breast', 'cocktail sauce', 'soup of the day carrot cream', 'curry dip', 'soup of the day lentil ginger', 'poulet cordon bleu', 'sliced seitan', 'mixed dish of poulet cordon bleu cauliflower soft egg noodles (spaetzle) and jus', 'mixed dish of sliced seitan pilaf rice and sugar peas', 'roasted cauliflower', 'pilau rice', 'soft egg noodles (spaetzle)', 'jus', 'sugar peas', 'sauce for sliced seitan', 'boiled potatoes', 'semolina porridge', 'cherry compote', 'cinnamon sugar', 'small plastic cup', 'oversoaked sliced chicken', 'mixed meal of semolina porridge and cherry compote', 'overly soft thick brie cheese', 'overly soft cottage cheese', 'mixed meal of meat cheese, mustard sauce and lyonnaise potatoes', 'oversoaked sliced veal', 'overly soft cream cheese', 'overly soft thin brie cheese', 'currants', 'lyonnaise potatoes', 'meat cheese', 'mustard sauce', 'soup of the day bell peppers', 'sliced quorn sauce zurich style', 'sliced quorn zurich style', 'colorful vegetables from zuchetti peas carrots and beans', 'hash brown (roesti)', 'bread dumplings', 'sauce poultry ragout', 'mixed dish of poultry ragout and bread dumplings', 'mixed meal of quorn vegetables and hash brown (roesti)', 'banana organic', 'croissant', 'lye croissant', 'coffee', 'jug lid on the ground', 'uncovered jug', 'jug covered with lid', 'orange juice', 'multigrain roll', 'scrambled eggs', 'milk', 'mueesli', 'large glass fruitsalad-bowl', 'milk roll', 'mozzarella', 'baked vegetables for mozzarella', 'wedges', 'chipolata sausage', 'vegetables for chipolata sausage', 'rocket', 'walnut', 'mixed meal of mozzarella salad, baked vegetables and house bread', 'vegetable piccata made from zuchetti eggplant and piccata mass', 'bramata slice', 'vegetables for piccata', 'cream', 'chocolate cake', 'fried rice', 'beef meatballs', 'spicy vegetable ragout', 'olives', 'cucumber', 'tomato', 'turkey breast', 'carrot', 'chili with vegetables', 'lenses brown', 'mixed dish of chili with vegetables lentils and romanesco', 'pear', 'apple', 'apricot tart', 'cheese crêpe', 'mixed meal of meatloaf peas mashed potatoes and sauce', 'soggy bread without crust', 'large square plate', 'oversoaked polenta', 'meatloaf', 'mashed potatoes', 'peas', 'soup of the day mushroom cream', 'cognac sauce', 'grated cheese', 'oversoaked chickpea curry', 'rice', 'oversoaked roast beef', 'oversoaked food in pyramid shape', 'swiss chard vegetable ragout', 'oversoaked chia pudding', 'oversoaked mixed roast beef', 'mixed dish of cream cheese sauce and swiss chard', 'oversoaked mixed chickpea curry', 'sardinian fregola', 'smoked sausage (landjaeger)', 'soup of the day leek cream', 'sardinian fregola with vegetables', 'vegetables for fregola', 'mustard', 'radish salad', 'fresh cheese praline', 'pickled cucumber', 'sliced quorn', 'soup of the day banana-coconut', 'bean cassoulet', 'salami', 'pickled vegetables', 'deli meat cheese', 'turkey', 'sour cream', 'cylindrical transparent shot glass', 'turkey ham', 'ricotta tortellini', 'potato vegetable curry', 'soup of the day sweetcorn', 'baked chickpea', 'cream sauce and tomato sauce', 'milk coffee', 'cherry jam', 'gruyere cheese', 'spreadable cheese', 'coffee cup', 'coffee plate', 'coffee yogurt', 'tilster cheese', 'brie cheese', 'margarine', 'appenzeller cheese', 'paprika sauce', 'bramata', 'green spelt dumplings', 'vegetable ragout for green spelt dumplings', 'chicken thigh steak', 'country cuts', 'veggie crispy bites', 'soup of the day tomatoes', 'vegetable salad for ham', 'country smoked ham', 'lye rolls', 'antipasti vegetables', 'tagliatelle tomato pesto antipasti', 'soup of the day beetroot', 'soy yogurt dip', 'vegan meatballs', 'vegetables for meatballs', 'yellow pea puree', 'boiled beef', 'horseradish bouillon', 'beef lasagna', 'eggplant cordon bleu', 'saffron risotto', 'bell pepper stew', 'pineapple-quark-mousse', 'tomato sauce', 'quinoa salad', 'dried tomatoes', 'endives orange salad', 'orange fillet', 'cashew nuts', 'creamy risotto', 'red onion', 'risotto', 'vegetable salad with quinoa', 'small glass bowl', 'vegetable salad for quinoa', 'thai glass noodle salad', 'cheese ravioli', 'fruit quark', 'pineapple', 'halibut', 'hummus', 'vegetables for halibut', 'asian dip', 'potato hash brown (roesti) with vegetables', 'oversoaked salmon fillet', 'oversoaked chickpea puree', 'bell pepper', 'port wine pears rucola risotto', 'soup of the day barley', 'rocket risotto', 'creamy polenta medium', 'beef patties in juicy sauce', 'merlot sauce', 'oversoaked perch fillet', 'oversoft food in crescent shape', 'mixed dish of rucola risotto gorgonzola and walnut', 'oversoaked mixed dish of salmon fillet, chickpea puree and bell peppers', 'oversoaked mixed dish of boiled beef polenta and carrots', 'mixed dish of beef patties in juicy sauce broccoli and polenta', 'oversoaked mixed dish of crescent-shaped perch fillets and zuchetti', 'potato herb patties', 'veggie cervalat sausage', 'colorful vegetables for veggie cervalat sausage', 'grisons barley soup', 'carrot puree', 'cod', 'ratatouille', 'soup of the day cauliflower cream', 'herbal semolina slice', 'crispy vegetable roll', 'rice noodle salad', 'soup of the day parmesan foam', 'creamed spinach', 'gnocchi pan tofu', 'homemade fishburgers', 'black bean puree', 'soup of the day zucchetti', 'roast beef', 'pizokel vegetable gratin', 'egg vinaigrette', 'cheesy soft egg noodles (kaesespaetzle)', 'cabbage salad', 'fennel salad for bowl', 'tree nut dressing', 'spelt marinated', 'feta marinated', 'beetroot cooked', 'iceberg lettuce', 'oversoaked smoked salmon', 'oversoaked mixed dish of smoked salmon zuchetti and panna cotta', 'mixed dish of protein bowl and multigrain roll', 'oversoaked fennel', 'oversoaked couscous', 'oversoaked turkey plate', 'lemon roulade', 'wild rice raw', 'homemade veggie burger', 'spicy tomato vegetable sauce', 'champignon organic', 'pork steak', 'gruyère', 'swiss macaroni and cheese', 'cantadou cheese', 'white bean puree', 'tilapia', 'burrito', 'pernod sauce', 'diced tomatoes', 'sliced beef', 'knot rolls', 'spelt goulash', 'french dressing', 'vegetables for salade nicoise consisting of beans spinach and lettuce', 'minced poultry patties', 'mixed dish of penne tofu and carbonara', 'carbonara tofu', 'rosemary sauce')
)
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
        num_classes=9,
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
    dict( # original configuration didn't have warmup step!
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False,  
        begin=0, 
        end=50),  # End at the 50th iteration # original: 250
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            22,
            24,
        ],
        type='MultiStepLR'),
]
resume = False

train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=None)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        ann_file='/opt/ml/input/data/train/instance_seg_train.json',
        backend_args=None,
        data_prefix=dict(img= '/opt/ml/input/data/train/images/'),
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
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))


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
