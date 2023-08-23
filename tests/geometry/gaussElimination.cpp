#include <geometry/utils.h>
#include <vector>
#include <cassert>
using namespace std;

int main(){
    vector<vector<double>> matrix = {
		{0, -4, 0, 1},
		{0, -1, 1, 1},
		{1, -2, 0, 1},
		{1, 1, 1, 1}
	};

    vector<double> ans = gaussElimination(matrix);

    assert(ans.size() == 3);
    assert(cmp(ans[0], 0.5) == 0);
    assert(cmp(ans[1], -0.25) == 0);
    assert(cmp(ans[2], 0.75) == 0);
}
