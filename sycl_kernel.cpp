#include 

using namespace sycl;

int main() {
    queue q;
    std::vector<int> data(1024, 1);
    buffer<int, 1> buf(data.data(), range<1>(data.size()));
    q.submit([&](handler &h) {
        auto acc = buf.get_access<access::mode::read_write>(h);
        h.parallel_for(range<1>(data.size()), [=](id<1> idx) {
            acc[idx] = idx[0] * 2;
        });
    }).wait();
    return 0;
}
