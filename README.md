Collaborative Filtering: From Loops to Latent Space

A compact, end-to-end MovieLens recommender built in a notebook.
We start with a loop-based cost function, switch to a vectorized version, add a brand-new user, normalize ratings, train with TensorFlow, and generate top movie predictions.

Why this is cool

The model doesn’t know genres or tags and it doesn’t even know the “axes” that matter.
By learning movie features X and user preferences W, it invents latent axes that explain ratings. Those dimensions capture taste without any labels.

What’s inside:
1. Loop cost: cofi_cost_func(...) with regularization
2. Vectorized cost: cofi_cost_func_vec(...) (same result, faster)
3. New user flow: add personal ratings, rebuild Y, R, and normalize
4. Training: TensorFlow GradientTape + Adam optimizer
5. Predictions: ranked recommendations for the new user

Key steps (at a glance)
1. Load precomputed parameters and ratings: load_precalc_params_small(), load_ratings_small().
2. Compute cost (loops), then vectorized cost; confirm they match.
3. Add a new user’s explicit ratings; prepend to Y and R.
4. Normalize with normalizeRatings(Y, R) (removes per-movie mean).
5. Train with TensorFlow (X, W, b are trainable).
6. Predict, restore means, sort, and print top recommendations.
