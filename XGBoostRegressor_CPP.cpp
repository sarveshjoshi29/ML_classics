// Creator : Sarvesh Joshi
// Date of creation: 14/7/2025
// Last updated: 14/7/2025

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

static constexpr int inf = 1e9;
using namespace std;

// For basic architecture refer python code. The only catch here is to define class Round as a public subclass of XGBoostRegressor and then define the
// private variable vector<Round*> rounds to store pointers to all rounds. This is initialized in the fit method.

template<typename T> class Node {

	public:
		int feature;
		double split_value;
		int depth;
		vector<double> Gradients;
		vector<double> Hessians;
		T data;
		Node<T>* left = nullptr;
		Node<T>* right = nullptr;
		bool is_leaf = false;

		// initialize node with constructor
		Node(int feature = -1, double split_value = (double)-1, int depth = 0, vector<double> Gradients = {-1}, vector<double> Hessians = {-1},
			 T data = T{})
			: feature{feature}, split_value{split_value}, depth{depth}, Gradients{Gradients}, Hessians{Hessians}, data{data} {

			is_leaf = (feature == -1) ? true : false;
		}

		// output function if node is a leaf
		double out(double lambda = 0.0, double alpha = 0.0) {
			if(is_leaf == true) {
				double G = 0.0;
				double H = 0.0;
				for(auto grad : Gradients) {
					G += grad;
				}
				for(auto hess : Hessians) {
					H += hess;
				}

				if(G > alpha) {
					return (double)(-(G - alpha) / (H + lambda));
				}

				else if(G < -alpha) {
					return (double)(-(G + alpha) / (H + lambda));
				}

				else {
					return (double)0.0;
				}
			}

			cerr << "Error : Out is called on a non-leaf node\n";
			exit(1);
		}
};

class Tree {
	public:
		Node<vector<vector<double>>>* root;
		Tree(Node<vector<vector<double>>>* root = nullptr) : root{root} {};
};

class XGBoostRegressor {

	private:
		int max_depth;
		string loss;
		double gamma;
		double lambda;
		double alpha;
		double learning_rate;
		int max_iter;
		int min_child_weight;
		static inline const string allowed_loss[1] = {"mse"};
		double subsample;

		// static method added for subasmpling data
		static tuple<vector<vector<double>>, vector<double>, vector<double>> subsample_data(const vector<vector<double>>& X, const vector<double>& y,
																							const vector<double>& y_guess, double ratio) {

			int req_size = (int)(X.size() * ratio);
			req_size = (req_size == 0) ? 1 : req_size;
			vector<int> idxs(X.size());
			iota(idxs.begin(), idxs.end(), 0);
			random_device rd;
			mt19937 gen(rd());
			shuffle(idxs.begin(), idxs.end(), gen);

			vector<int> selected_idxs;
			for(int i = 0; i < req_size; i++) {
				selected_idxs.push_back(idxs[i]);
			}

			vector<vector<double>> X_sampled;
			vector<double> y_sampled;
			vector<double> y_guess_sampled;
			for(int i = 0; i < selected_idxs.size(); i++) {
				X_sampled.push_back(X[selected_idxs[i]]);
				y_sampled.push_back(y[selected_idxs[i]]);
				y_guess_sampled.push_back(y_guess[selected_idxs[i]]);
			}

			return make_tuple(X_sampled, y_sampled, y_guess_sampled);
		}

	public:
		// init xgboostregressor constructor
		XGBoostRegressor(int max_depth = 50, string loss = "mse", double gamma = 0.0, double lambda = 0.0, double alpha = 0.0,
						 double learning_rate = 0.1, int max_iter = 50, int min_child_weight = 0, double subsample = 1.0)
			: max_depth{max_depth}, loss{loss}, gamma{gamma}, lambda{lambda}, alpha{alpha}, learning_rate{learning_rate}, max_iter{max_iter},
			  min_child_weight{min_child_weight}, subsample{subsample}

		{
			// convert loss to lowercase and check if it is in allowed losses else raise ValueError
			transform(this->loss.begin(), this->loss.end(), this->loss.begin(), ::tolower);

			bool found_allowed_loss = false;

			for(auto key : this->allowed_loss) {
				found_allowed_loss = (key == loss) ? true : found_allowed_loss;
			}
			if(!found_allowed_loss) {
				cerr << " ValueError : Loss can only be mse\n";
				exit(1);
			}

			if(this->subsample > 1 || this->subsample <= 0) {
				cerr << "ValueError: subsample must be between 0 and 1\n";
				exit(1);
			}
		}

		class Round {
			private:
				Tree* tree = nullptr;
				int round_num;
				vector<double> Gradients = {};
				vector<double> Hessians = {};
				vector<Node<vector<vector<double>>>*> leaves;
				vector<Node<vector<vector<double>>>*> nodes;
				XGBoostRegressor* model;

			public:
				// init constructor for class Round
				Tree* get_tree() {
					return tree;
				}

				Round(int round_num = 0, XGBoostRegressor* model = nullptr) : round_num{round_num}, model{model} {
				}

				// compute grads based on loss function
				void compute_Grads(vector<double> y_train, vector<double> y_guess) {
					Gradients.clear();
					Hessians.clear();
					if(model->loss == "mse") {
						for(int i = 0; i < y_train.size(); i++) {

							Gradients.push_back(y_guess[i] - y_train[i]);
							Hessians.push_back((double)1);
						}
						return;
					}
				}

				// brute force through all posibilities to find optimal feature and split value
				pair<int, double> find_split_value(const vector<vector<double>>& X, const vector<double>& grads_node,
												   const vector<double>& hess_node) {

					int feature = -1;
					double split_value = (double)0;
					double G = 0.0;
					double H = 0.0;
					for(int i = 0; i < grads_node.size(); i++) {
						G += grads_node[i];
						H += hess_node[i];
					}

					double parent_gain = 0.5 * ((pow(G, 2)) / (H + this->model->lambda));

					double max_gain = -inf;

					for(int curr_feature = 0; curr_feature < X[0].size(); curr_feature++) {

						// idea is to create a index mask sorted on the basis of curr_col

						vector<int> index_mask(X.size());
						iota(index_mask.begin(), index_mask.end(), 0);

						vector<double> curr_col;
						for(int row = 0; row < X.size(); row++) {
							curr_col.push_back(X[row][curr_feature]);
						}

						sort(index_mask.begin(), index_mask.end(), [&](int a, int b) -> bool { return curr_col[a] < curr_col[b]; });
						vector<double> grads_sorted;
						vector<double> hess_sorted;
						vector<double> curr_col_sorted;
						for(auto idx : index_mask) {
							grads_sorted.push_back(grads_node[idx]);
							hess_sorted.push_back(hess_node[idx]);
							curr_col_sorted.push_back(curr_col[idx]);
						}

						double G_left{};
						double H_left{};
						for(int row = 0; row < curr_col_sorted.size() - 1; row++) {

							G_left += grads_sorted[row];
							H_left += hess_sorted[row];

							if(curr_col_sorted[row] == curr_col_sorted[row + 1]) {
								continue;
							}

							double curr_split_val = (curr_col_sorted[row] + curr_col_sorted[row + 1]) / 2;
							double G_right = G - G_left;
							double H_right = H - H_left;

							if(H_left < this->model->min_child_weight || H_right < this->model->min_child_weight) {
								continue;
							}

							double curr_gain = 0.5 * ((pow(G_left, 2)) / (H_left + this->model->lambda) +
													  (pow(G_right, 2)) / (H_right + this->model->lambda) - parent_gain) -
											   this->model->gamma;

							if(curr_gain > 0 && curr_gain > max_gain) {
								feature = curr_feature;
								split_value = curr_split_val;
								max_gain = curr_gain;
							}
						}
					}

					return make_pair(feature, split_value);
				}

				//----------------------------------------------------------

				Node<vector<vector<double>>>* create_node(const vector<vector<double>>& data, const int depth, const vector<double>& grads_node,
														  const vector<double>& hess_node) {
					pair<int, double> info = find_split_value(data, grads_node, hess_node);
					Node<vector<vector<double>>>* new_node = new Node(info.first, info.second, depth, grads_node, hess_node, data);
					this->nodes.push_back(new_node);
					return new_node;
				}

				// build tree using queue (like bfs)
				void build_tree(const vector<vector<double>>& X) {
					int curr_depth = 1;
					Node<vector<vector<double>>>* root = create_node(X, curr_depth, this->Gradients, this->Hessians);
					this->tree = new Tree(root);
					queue<Node<vector<vector<double>>>*> q;
					q.push(root);

					while(!q.empty()) {
						Node<vector<vector<double>>>* curr_node = q.front();
						q.pop();
						if(curr_node->depth >= this->model->max_depth) {
							curr_node->is_leaf = true;
							leaves.push_back(curr_node);
							continue;
						}
						if(!(curr_node->is_leaf)) {

							int curr_feature = curr_node->feature;
							double curr_split_value = curr_node->split_value;
							vector<double> curr_grads = curr_node->Gradients;
							vector<double> curr_hess = curr_node->Hessians;
							int curr_depth = curr_node->depth;

							vector<int> index_left;
							vector<int> index_right;

							// creating index masks for split data
							for(int i = 0; i < X.size(); i++) {
								if(X[i][curr_feature] < curr_split_value) {
									index_left.push_back(i);
								} else {
									index_right.push_back(i);
								}
							}

							if(index_left.size() > 0 && index_right.size() > 0) {
								vector<vector<double>> data_left;
								vector<vector<double>> data_right;
								vector<double> grads_left;
								vector<double> grads_right;
								vector<double> hess_left;
								vector<double> hess_right;

								for(auto idx : index_left) {
									data_left.push_back(X[idx]);
									grads_left.push_back(curr_grads[idx]);
									hess_left.push_back(curr_hess[idx]);
								}
								for(auto idx : index_right) {
									data_right.push_back(X[idx]);
									grads_right.push_back(curr_grads[idx]);
									hess_right.push_back(curr_hess[idx]);
								}

								curr_node->left = create_node(data_left, curr_depth + 1, grads_left, hess_left);
								curr_node->right = create_node(data_right, curr_depth + 1, grads_right, hess_right);
								q.push(curr_node->left);
								q.push(curr_node->right);
							}

							else {
								leaves.push_back(curr_node);
							}
						}

						else {
							leaves.push_back(curr_node);
						}
					}
				}

				//----------------------------------------------------------------------------

				void evaluate_tree(const vector<vector<double>>& X, vector<double>& y_guess) {
					if(this->tree->root == nullptr) {
						cerr << "Tree is uninitialized. Call fit before transform";
						exit(1);
					}
					// traverse the tree per sample and get output from whichever leaf it reaches
					for(int i = 0; i < X.size(); i++) {
						vector<double> curr_row = X[i];
						Node<vector<vector<double>>>* curr_node = tree->root;
						while(!curr_node->is_leaf) {
							if(curr_row[curr_node->feature] < curr_node->split_value) {
								curr_node = curr_node->left;
							} else {
								curr_node = curr_node->right;
							}
						}

						double out_leaf = curr_node->out(this->model->lambda, this->model->alpha);
						y_guess[i] += this->model->learning_rate * out_leaf;
					}
				}
		};

	private:
		vector<Round*> rounds;

	public:
		Round* create_round(int round_num) {
			Round* round = new Round(round_num, this);
			return round;
		}

		void fit(vector<vector<double>>& X_train, vector<double>& y_train) {
			vector<double> y_guess(y_train.size(), (double)0);

			this->rounds.clear();
			// Very Imp because same instance can be fitted on multiple data.

			for(int i = 0; i < this->max_iter; i++) {
				this->rounds.push_back(create_round(i));
			}

			for(auto round : rounds) {

				tuple<vector<vector<double>>, vector<double>, vector<double>> sampled_data = subsample_data(X_train, y_train, y_guess, subsample);
				round->compute_Grads(get<1>(sampled_data), get<2>(sampled_data));
				round->build_tree(get<0>(sampled_data));
				round->evaluate_tree(X_train, y_guess);
			}
		}

		vector<double> predict(vector<vector<double>> X_test) {
			vector<double> y_guess(X_test.size(), (double)0);
			if(rounds.empty()) {
				cerr << "ValueError : Rounds are not initialized. Please call fit before predict\n";
				exit(1);
			}
			// int ct{};
			for(auto round : rounds) {
				round->evaluate_tree(X_test, y_guess);
				if(round->get_tree()->root->left != nullptr) {
					// ct++;
				}
			}
			// cout << "Number of trees used is " << ct << "\n";
			return y_guess;
		}
};

int main() {
	vector<vector<double>> X_train = {{1.5, 2.0}, {2.1, 1.0}, {3.7, 4.2}, {4.1, 3.6}, {5.9, 5.2}, {6.0, 7.0}};

	// y = 2x[0] + 4x[1] +- noise;
	vector<double> y_train;
	mt19937 gen(29);
	uniform_real_distribution<> dis(-0.25, 0.25);
	for(int i = 0; i < X_train.size(); i++) {
		double noise = dis(gen);
		y_train.push_back(2 * X_train[i][0] + 4 * X_train[i][1] + noise);
	}

	XGBoostRegressor model(30,	  // max_depth
						   "mse", // loss
						   0.0,	  // gamma
						   0.0,	  // lambda
						   0.0,	  // alpha
						   0.6,	  // learning rate
						   50,	  // max_iter
						   0,	  // min_child_weight
						   1.0	  // subsample
	);

	model.fit(X_train, y_train);

	vector<double> preds = model.predict(X_train);

	cout << "Predictions:" << endl;
	for(double p : preds) {
		cout << p << endl;
	}
	cout << "\n\n\n";

	cout << "Actual:" << endl;
	for(double y : y_train) {
		cout << y << endl;
	}

	return 0;
}
