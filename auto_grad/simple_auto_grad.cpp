#include<iostream>
#include<vector>
using namespace std;
//auto grad project pytorch simulation

class tensor {
	enum fun { none, sum, mul };

	float data;
	float gradient = 0;
	enum fun f;
	vector<tensor*> parent;
	tensor(float data, tensor* parnet1 = nullptr, tensor* parent2 = nullptr) {
		this->data = data;
		if (parnet1 != nullptr)parent.push_back(parnet1);
		if (parent2 != nullptr)parent.push_back(parent2);
	}

public:
	void set_gredient(float gradient) {
		this->gradient += gradient;
	}
	float get_gradient()
	{
		return gradient;
	}
	float get_data()
	{
		return data;
	}
	tensor* create_tensor(float data, tensor* parnet1 = nullptr, tensor* parent2 = nullptr)
	{
		return new tensor(data, parnet1, parent2);
	}
	tensor() {

	}
	void print_tensor() {
		cout << "the data filed is " << data << " the gradient is " << gradient 
			<< "and the parent is " << parent.back() << "\t" << parent.back() - 1;
	}
	tensor* add(tensor* t1) {
		tensor * t = new tensor(t1->data + this->data, this, t1);
		t->f = sum;
		return t;
	}
	tensor* multiply(tensor* t1) {
		tensor* t = new tensor(t1->data * this->data, this, t1);
		t->f = mul;
		return t;
	}
	void backward() {
		if (parent.empty()) return;

		tensor* t1 = parent[0];
		tensor* t2 = parent[1];

		if (this->f == sum) {
			t1->set_gredient(1.0 * this->gradient);
			t2->set_gredient(1.0 * this->gradient);
		}
		else if (this->f == mul) {
			t1->set_gredient(t2->get_data() * this->gradient);
			t2->set_gredient(t1->get_data() * this->gradient);
		}

		t1->backward();
		t2->backward();
	}
};

int main()
{
	tensor obj;

	cout << "=== Test 1: f = (x * y) + z ===" << endl;
	auto x = obj.create_tensor(2);
	auto y = obj.create_tensor(3);
	auto z = obj.create_tensor(4);

	auto a = x->multiply(y);
	auto f1 = a->add(z);    

	f1->set_gredient(1.0);
	f1->backward();

	cout << "x gradient (Should be 3): " << x->get_gradient() << endl;
	cout << "y gradient (Should be 2): " << y->get_gradient() << endl;
	cout << "z gradient (Should be 1): " << z->get_gradient() << endl;
	cout << "-----------------------------------" << endl;

	cout << "\n=== Test 2: The Topology Trap (f = x * x) ===" << endl;
	auto x2 = obj.create_tensor(3);
	auto f2 = x2->multiply(x2); 

	f2->set_gredient(1.0);
	f2->backward();

	cout << "x2 gradient (Should be 6): " << x2->get_gradient() << endl;
}
