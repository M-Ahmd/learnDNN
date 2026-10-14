#include<iostream>
#include<vector>
#include<unordered_set>
#include<stack>
using namespace std;

class tensor;

class gradiable {
public:
    virtual void grad() = 0;
};

class sum : public gradiable {
    tensor* p1;
    tensor* p2;
    tensor* out;
public:
    sum(tensor* p1, tensor* p2, tensor* out) {
        this->p1 = p1;
        this->p2 = p2;
        this->out = out;
    }
    void grad() override; 
};

class mul : public gradiable {
    tensor* p1;
    tensor* p2;
    tensor* out;
public:
    mul(tensor* p1, tensor* p2, tensor* out) {
        this->p1 = p1;
        this->p2 = p2;
        this->out = out;
    }
    void grad() override; 
};

class tensor {
    float data;
    gradiable* function = nullptr; 
    float gradient = 0;
    vector<tensor*> parent;

    void build_graph(unordered_set<tensor*>& visted, stack<tensor*>& q, tensor* t) {
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
public:
    void set_gredient(float gradient) { this->gradient += gradient; }
    float get_gradient() { return gradient; }
    float get_data() { return data; }

    tensor(float data, tensor* parnet1 = nullptr, tensor* parent2 = nullptr) {
        this->data = data;
        if (parnet1 != nullptr) parent.push_back(parnet1);
        if (parent2 != nullptr) parent.push_back(parent2);
    }
    tensor() {}

    tensor* create_tensor(float data, tensor* parnet1 = nullptr, tensor* parent2 = nullptr) {
        return new tensor(data, parnet1, parent2);
    }

    void print_tensor() {
        cout << "the data filed is " << data << " the gradient is " << gradient << endl;
    }

    tensor* add(tensor* t1) {
        tensor* t = new tensor(t1->data + this->data, this, t1);
        t->function = new sum(this, t1, t);
        return t;
    }

    tensor* multiply(tensor* t1) {
        tensor* t = new tensor(t1->data * this->data, this, t1);
        t->function = new mul(this, t1, t);
        return t;
    }

    void backward() {
        unordered_set<tensor*> set;
        stack<tensor*> q;
        build_graph(set, q, this);

        this->gradient = 1.0;

        while (!q.empty()) {
            q.top()->_step_backward();
            q.pop();
        }
    }
};

void sum::grad() {
    p1->set_gredient(1.0 * out->get_gradient());
    p2->set_gredient(1.0 * out->get_gradient());
}

void mul::grad() {
    p1->set_gredient(p2->get_data() * out->get_gradient());
    p2->set_gredient(p1->get_data() * out->get_gradient());
}

int main()
{
    tensor obj;

    cout << "=== Test 1: f = (x * y) + z ===" << endl;
    auto x = obj.create_tensor(2);
    auto y = obj.create_tensor(3);
    auto z = obj.create_tensor(4);

    auto a = x->multiply(y);
    auto f1 = a->add(z);

    f1->backward(); 

    cout << "x gradient (Should be 3): " << x->get_gradient() << endl;
    cout << "y gradient (Should be 2): " << y->get_gradient() << endl;
    cout << "z gradient (Should be 1): " << z->get_gradient() << endl;
    cout << "-----------------------------------" << endl;

    cout << "\n=== Test 2: The Topology Trap (f = x * x) ===" << endl;
    auto x2 = obj.create_tensor(3);
    auto f2 = x2->multiply(x2);

    f2->backward();

    cout << "x2 gradient (Should be 6): " << x2->get_gradient() << endl;
}