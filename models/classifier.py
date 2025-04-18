import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier

class XGBoostClassifier:
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
###########################################################################

class GLU_Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        x_proj = self.fc(x)
        x_a, x_b = x_proj.chunk(2, dim=-1)
        return x_a * torch.sigmoid(x_b)


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.block1 = GLU_Block(input_dim, hidden_dim)
        self.block2 = GLU_Block(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, prior):
        x = self.fc(x)
        x = self.bn(x)
        mask = torch.softmax(x * prior, dim=-1)
        return mask


class TabNet(nn.Module):
    def __init__(self, config):
        super(TabNet, self).__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.n_steps = config.get('n_steps', 3)
        self.num_classes = config['num_classes']
        
        self.initial_transform = FeatureTransformer(self.input_dim, self.hidden_dim)
        self.attentive = nn.ModuleList([
            AttentiveTransformer(self.hidden_dim, self.input_dim) for _ in range(self.n_steps)
        ])
        self.transformers = nn.ModuleList([
            FeatureTransformer(self.input_dim, self.hidden_dim) for _ in range(self.n_steps)
        ])
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x):
        out_agg = 0
        prior = torch.ones_like(x)
        x_raw = x.clone()  
        
        x_transformed = self.initial_transform(x_raw)
        
        for step in range(self.n_steps):
            mask = self.attentive[step](x_transformed, prior)  
            x_masked = x_raw * mask 
            step_output = self.transformers[step](x_masked)
            out_agg = out_agg + step_output
            prior = prior * (1 - mask)

        logits = self.fc(out_agg)
        return F.log_softmax(logits, dim=-1)

###########################################################################

class NODEBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_trees, depth=2):
        super(NODEBlock, self).__init__()
        self.num_trees = num_trees
        self.depth = depth
        self.hidden_dim = hidden_dim

        # Oblivious Decision Tree
        self.feature_selection = nn.Linear(input_dim, num_trees)
        self.thresholds = nn.Parameter(torch.randn(num_trees, depth))

        self.num_leaves = num_trees * (2 ** depth)  
        self.leaf_weights = nn.Linear(self.num_leaves, hidden_dim)

    def forward(self, x):
        feature_scores = self.feature_selection(x)  # (batch_size, num_trees)
        feature_decisions = torch.sigmoid(feature_scores.unsqueeze(-1) - self.thresholds)  # (batch_size, num_trees, depth)

        leaves = feature_decisions.prod(dim=-1)  # (batch_size, num_trees)
        leaves = torch.cat([leaves, 1 - leaves], dim=-1)  # (batch_size, num_trees * 2)

        leaves = leaves.repeat(1, 2 ** (self.depth - 1))  # (batch_size, num_leaves)

        out = self.leaf_weights(leaves)  # (batch_size, hidden_dim)
        return out

class NODE(nn.Module):
    def __init__(self, input_dim, config):
        super(NODE, self).__init__()
        self.node_block = NODEBlock(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_trees=config['num_trees'],
            depth=config.get('depth', 2)
        )
        self.fc = nn.Linear(config['hidden_dim'], config['num_classes'])

    def forward(self, x):
        x = self.node_block(x)
        x = self.fc(x)
        return F.softmax(x, dim=-1)