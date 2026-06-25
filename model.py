import torch
import torch.nn as nn
import numpy as np
import pysindy as ps

class FCEncoder(nn.Module):
    def __init__(self, params):
        super(FCEncoder, self).__init__()
        
        input_dim = params['input_dim']
        output_dim = params['latent_dim']
        hidden_dims = params.get('hidden_dims', []) 
        activation_name = params.get('non_linear', "")
        
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU()
        }
        
        activation = None
        if activation_name:
            activation = activations.get(activation_name.lower())
            if activation is None:
                raise ValueError(f"Unsupported activation: {activation_name}")

        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if activation:
                layers.append(activation)
            current_dim = h_dim 
            
        # Final output layer mapping to the latent space z (no activation)
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class FCDecoder(nn.Module):
    def __init__(self, params):
        super(FCDecoder, self).__init__()
        
        latent_dim = params['latent_dim']
        output_dim = params['input_dim']
        hidden_dims = params.get('hidden_dims', [])
        
        # Reverse the hidden layers to expand back outward
        decoder_hidden_dims = hidden_dims[::-1] 
        activation_name = params.get('non_linear', "")
        
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU()
        }
        
        activation = None
        if activation_name:
            activation = activations.get(activation_name.lower())

        layers = []
        current_dim = latent_dim
        
        for h_dim in decoder_hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if activation:
                layers.append(activation)
            current_dim = h_dim 
            
        # Final output layer mapping back to the high-dimensional space (no activation)
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.network(z)

class FastSINDyLibraryLayer(nn.Module):
    # Added 'params' to the signature
    def __init__(self, library_input_dim, output_dim, params, poly_degree=2, include_bias=True, add_sine=False):
        super(FastSINDyLibraryLayer, self).__init__()
        self.library_input_dim = library_input_dim
        self.output_dim = output_dim
        self.add_sine = add_sine
        
        # 1. Use PySINDy for the polynomial layout based on the input dimension
        poly_library = ps.PolynomialLibrary(degree=poly_degree, include_bias=include_bias)
        poly_library.fit(np.zeros((1, library_input_dim)))
        self.feature_names = poly_library.get_feature_names()
        
        num_poly_features = len(self.feature_names)
        power_matrix = torch.zeros(num_poly_features, library_input_dim)
        
        for i, name in enumerate(self.feature_names):
            if name == '1': 
                continue
            for factor in name.split(' '):
                if '^' in factor:
                    var, power = factor.split('^')
                    power_matrix[i, int(var[1:])] = float(power)
                else:
                    power_matrix[i, int(factor[1:])] = 1.0
                    
        self.register_buffer('power_matrix', power_matrix)
        
        # 2. Add Sine/Cosine tracking if requested
        if self.add_sine:
            for i in range(library_input_dim):
                self.feature_names.append(f"sin(x{i})")
            for i in range(library_input_dim):
                self.feature_names.append(f"cos(x{i})")
                
        # 3. Initialize trainable coefficient matrix Xi
        total_features = len(self.feature_names)
        self.xi = nn.Parameter(torch.empty(total_features, output_dim))
        
        init_type = params.get('coefficient_initialization', 'normal')
        
        if init_type == 'xavier':
            nn.init.xavier_uniform_(self.xi)
        elif init_type == 'constant':
            nn.init.constant_(self.xi, 1.0)
        elif init_type == 'normal':
            nn.init.normal_(self.xi, mean=0.0, std=0.1) 
        elif init_type == 'specified':
            init_coeffs = torch.tensor(params['init_coefficients'], dtype=torch.float32)
            self.xi.data.copy_(init_coeffs)
            
        # REMOVED the duplicate self.xi assignment that was overriding the logic above!

    def forward(self, features, mask=None):
        features_expanded = features.unsqueeze(1) 
        features_pow = features_expanded ** self.power_matrix
        Theta_poly = torch.prod(features_pow, dim=2) 
        
        if self.add_sine:
            Theta_sin = torch.sin(features)
            Theta_cos = torch.cos(features)
            Theta = torch.cat([Theta_poly, Theta_sin, Theta_cos], dim=1)
        else:
            Theta = Theta_poly
        
        # Apply the mask to Xi BEFORE predicting the dynamics
        if mask is not None:
            xi_effective = self.xi * mask
        else:
            xi_effective = self.xi
            
        prediction = torch.matmul(Theta, xi_effective)
        return prediction, Theta
   
class SINDyAutoencoder(nn.Module):
    def __init__(self, params):  # Ensure it doesn't default to poly_degree=2 here
        super(SINDyAutoencoder, self).__init__()
        
        self.model_order = params.get('model_order', 1)
        self.encoder = FCEncoder(params)
        self.decoder = FCDecoder(params)
        
        latent_dim = params['latent_dim']
        library_input_dim = latent_dim if self.model_order == 1 else 2 * latent_dim
        
        # 1. READ THE POLY ORDER FROM PARAMS
        poly_degree = params.get('poly_order', 3) 
        
        self.sindy_layer = FastSINDyLibraryLayer(
            library_input_dim=library_input_dim, 
            output_dim=latent_dim,
            poly_degree=poly_degree, # 2. PASS IT TO THE LAYER
            include_bias=True,
            add_sine=params.get('include_sine', False),
            params=params
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)


    def forward(self, x, dx, ddx=None, mask=None):
        network = {}
        network['x'] = x
        network['dx'] = dx
        
        if self.model_order == 1:
            # 1. Exact First Derivatives: z = encoder(x), dz = Jacobian * dx
            z, dz = torch.autograd.functional.jvp(
                self.encoder, x, v=dx, create_graph=True
            )
            
            # 2. Predict Latent Dynamics
            dz_predict, Theta = self.sindy_layer(z, mask=mask)
            
            # 3. Decode & Propagate Derivatives (dx_decode from dz_predict)
            x_decode, dx_decode = torch.autograd.functional.jvp(
                self.decoder, z, v=dz_predict, create_graph=True
            )
            
            network.update({
                'z': z, 'dz': dz, 'x_decode': x_decode, 'dx_decode': dx_decode,
                'Theta': Theta, 'dz_predict': dz_predict, 
                'sindy_coefficients': self.sindy_layer.xi
            })
            
        elif self.model_order == 2:
            if ddx is None:
                raise ValueError("ddx must be provided for model_order=2")
            network['ddx'] = ddx
            
            # 1. First Derivatives
            z, dz = torch.autograd.functional.jvp(
                self.encoder, x, v=dx, create_graph=True
            )
            
            # 2. Exact Second Derivatives (jvp of the jvp)
            def get_dz(x_in, dx_in):
                return torch.autograd.functional.jvp(self.encoder, x_in, v=dx_in, create_graph=True)[1]
                
            _, ddz = torch.autograd.functional.jvp(
                get_dz, (x, dx), v=(dx, ddx), create_graph=True
            )
            
            # 3. Predict Latent Dynamics (library takes both z and dz)
            z_combined = torch.cat([z, dz], dim=1)
            ddz_predict, Theta = self.sindy_layer(z_combined, mask=mask)
            
            # 4. Decode First Order
            x_decode, dx_decode = torch.autograd.functional.jvp(
                self.decoder, z, v=dz, create_graph=True
            )
            
            # 5. Decode Second Order based on the prediction (ddz_predict)
            def get_dx_decode(z_in, dz_in):
                return torch.autograd.functional.jvp(self.decoder, z_in, v=dz_in, create_graph=True)[1]
                
            _, ddx_decode = torch.autograd.functional.jvp(
                get_dx_decode, (z, dz), v=(dz, ddz_predict), create_graph=True
            )
            
            network.update({
                'z': z, 'dz': dz, 'ddz': ddz,
                'x_decode': x_decode, 'dx_decode': dx_decode, 'ddx_decode': ddx_decode,
                'Theta': Theta, 'ddz_predict': ddz_predict, 
                'sindy_coefficients': self.sindy_layer.xi
            })

        return network


def define_loss(network, params):
    """
    Create the PyTorch loss functions.

    Arguments:
        network - Dictionary containing the elements of the network architecture.
                  (Output of the SINDyAutoencoder's forward() pass)
        params  - Dictionary containing loss weights, model order, and coefficient mask.
    """
    x = network['x']
    x_decode = network['x_decode']
    
    # Handle the coefficient mask for sequential thresholding
    sindy_coefficients = network['sindy_coefficients']
    if 'coefficient_mask' in params and params['coefficient_mask'] is not None:
        mask = params['coefficient_mask']
        # Ensure the mask is a PyTorch tensor on the same device as the network
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=sindy_coefficients.dtype, device=sindy_coefficients.device)
        masked_coefficients = mask * sindy_coefficients
    else:
        masked_coefficients = sindy_coefficients

    losses = {}
    
    # 1. Decoder Reconstruction Loss
    losses['decoder'] = torch.mean((x - x_decode)**2)

    # 2. SINDy Latent & Input Losses (Handles Order 1 vs Order 2)
    if params.get('model_order', 1) == 1:
        dz = network['dz']
        dz_predict = network['dz_predict']
        dx = network['dx']
        dx_decode = network['dx_decode']
        
        losses['sindy_z'] = torch.mean((dz - dz_predict)**2)
        losses['sindy_x'] = torch.mean((dx - dx_decode)**2)
    else:
        ddz = network['ddz']
        ddz_predict = network['ddz_predict']
        ddx = network['ddx']
        ddx_decode = network['ddx_decode']
        
        losses['sindy_z'] = torch.mean((ddz - ddz_predict)**2)
        losses['sindy_x'] = torch.mean((ddx - ddx_decode)**2)

    # 3. SINDy Regularization (L1 norm)
    losses['sindy_regularization'] = torch.mean(torch.abs(masked_coefficients))

    # 4. Total Loss Combinations
    loss = (params['loss_weight_decoder'] * losses['decoder'] +
            params['loss_weight_sindy_z'] * losses['sindy_z'] +
            params['loss_weight_sindy_x'] * losses['sindy_x'] +
            params['loss_weight_sindy_regularization'] * losses['sindy_regularization'])

    # Refinement loss removes the L1 penalty (used after the sparsity mask is fixed)
    loss_refinement = (params['loss_weight_decoder'] * losses['decoder'] +
                       params['loss_weight_sindy_z'] * losses['sindy_z'] +
                       params['loss_weight_sindy_x'] * losses['sindy_x'])

    return loss, losses, loss_refinement