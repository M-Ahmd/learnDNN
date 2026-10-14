#include<iostream>
#include<vector>
#include<unordered_set>
#include<stack>
#include<memory> 

using namespace std;

class tensor;

class gradiable {
public:
    virtual void grad() = 0;
    virtual ~gradiable() = default;
};

class sum_backward : public gradiable {
    shared_ptr<tensor> p1, p2, out;
public:
    sum_backward(shared_ptr<tensor> p1, shared_ptr<tensor> p2, shared_ptr<tensor> out)
        : p1(p1), p2(p2), out(out) {
    }
    void grad() override;
};

class mul_backward : public gradiable {
    shared_ptr<tensor> p1, p2, out;
public:
    mul_backward(shared_ptr<tensor> p1, shared_ptr<tensor> p2, shared_ptr<tensor> out)
        : p1(p1), p2(p2), out(out) {
    }
    void grad() override;
};

class tensor : public std::enable_shared_from_this<tensor> {
    float data;
    unique_ptr<gradiable> function = nullptr;
    float gradient = 0;
    vector<shared_ptr<tensor>> parent; 

    void build_graph(unordered_set<shared_ptr<tensor>>& visted, stack<shared_ptr<tensor>>& q, shared_ptr<tensor> t) {
        if (visted.find(t) == visted.end()) {
            visted.insert(t);
            for (auto& p : t->parent) {
                build_graph(visted, q, p);
            }
            q.push(t);
        }
    }

    void _step_backward() {
        if (function != nullptr) {
            function->grad();
        }
    }

    void build_zero_grad(unordered_set<shared_ptr<tensor>>& visted, shared_ptr<tensor> t) {
        if (visted.find(t) == visted.end()) {
            visted.insert(t);
            t->gradient = 0; 
            for (auto& p : t->parent) {
                build_zero_grad(visted, p);
            }
        }
    }

public:
    void set_gredient(float gradient) { this->gradient += gradient; }
    float get_gradient() { return gradient; }
    float get_data() { return data; }

    tensor(float data) : data(data) {}
    tensor() {}

    static shared_ptr<tensor> create_tensor(float data) {
        return make_shared<tensor>(data);
    }

    void print_tensor() {
        cout << "Data: " << data << " | Grad: " << gradient << endl;
    }

    shared_ptr<tensor> add(shared_ptr<tensor> t1) {
        auto t = make_shared<tensor>(t1->data + this->data);
        t->parent.push_back(shared_from_this()); 
        t->parent.push_back(t1);
        t->function = make_unique<sum_backward>(shared_from_this(), t1, t);
        return t;
    }

    shared_ptr<tensor> multiply(shared_ptr<tensor> t1) {
        auto t = make_shared<tensor>(t1->data * this->data);
        t->parent.push_back(shared_from_this());
        t->parent.push_back(t1);
        t->function = make_unique<mul_backward>(shared_from_this(), t1, t);
        return t;
    }

    void backward() {
        unordered_set<shared_ptr<tensor>> set;
        stack<shared_ptr<tensor>> q;
        build_graph(set, q, shared_from_this());

        this->gradient = 1.0;

        while (!q.empty()) {
            q.top()->_step_backward();
            q.pop();
        }
    }

    void zero_grad() {
        unordered_set<shared_ptr<tensor>> visited;
        build_zero_grad(visited, shared_from_this());
    }
};

void sum_backward::grad() {
    p1->set_gredient(1.0 * out->get_gradient());
    p2->set_gredient(1.0 * out->get_gradient());
}

void mul_backward::grad() {
    p1->set_gredient(p2->get_data() * out->get_gradient());
    p2->set_gredient(p1->get_data() * out->get_gradient());
}

int main() {
    cout << "=== The Final Engine Test ===" << endl;
    auto x = tensor::create_tensor(2);
    auto y = tensor::create_tensor(3);
    auto z = tensor::create_tensor(4);

    auto a = x->multiply(y);
    auto f1 = a->add(z);

    f1->backward();
    cout << "--- First Backward ---" << endl;
    cout << "x grad (3): " << x->get_gradient() << endl;
    cout << "y grad (2): " << y->get_gradient() << endl;
    cout << "z grad (1): " << z->get_gradient() << endl;

    f1->zero_grad();
    cout << "\n--- After zero_grad() ---" << endl;
    cout << "x grad (0): " << x->get_gradient() << endl;

    f1->backward();
    cout << "\n--- Second Backward ---" << endl;
    cout << "x grad (3): " << x->get_gradient() << endl;
}