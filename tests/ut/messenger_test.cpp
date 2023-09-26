#include "messenger.h"

#include "gtest/gtest.h"

class MyTestSuite : public ::testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    static Messenger myMessenger;
};

Messenger MyTestSuite::myMessenger;

TEST_F(MyTestSuite, broadcast) {
    int ids[128];
    if (myMessenger.isMaster()) {
        for (int i = 0; i < sizeof(ids) / sizeof(ids[0]); ++i) {
            ids[i] = i + 1;
        }
    }

    myMessenger.broadcast(ids, sizeof(ids) / sizeof(ids[0]));

    for (int i = 0; i < sizeof(ids) / sizeof(ids[0]); ++i) {
        EXPECT_EQ(ids[i], i + 1);
    }
}

TEST_F(MyTestSuite, allreduce) {
    float data[128];
    for (int i = 0; i < sizeof(data) / sizeof(data[0]); ++i) {
        data[i] = myMessenger.getRank() + 1;
    }

    myMessenger.reduceAdd(data, data, sizeof(data) / sizeof(data[0]));

    int size = myMessenger.getSize();
    float expected = size * (size + 1) / 2;
    float eps = 1e-5;

    for (int i = 0; i < sizeof(data) / sizeof(data[0]); ++i) {
        EXPECT_NEAR(data[i], expected, eps);
    }
}

int main(int argc, char **argv) {
    // Use system clock for seed
    srand(time(NULL));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
