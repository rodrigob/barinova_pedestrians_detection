#pragma once

#define _CRTDBG_MAP_ALLOC

#define VOTE_UPDATES

#include <cstdlib>

#ifdef WIN32
#include <crtdbg.h> // windows specific header
#endif

#include <cstdio>
#include <cassert>

const double BAD_SPLIT = -1E30;

template<class Sample, class Test, typename Vote>
class RandomTree
{
public:
    //tree nodes
    struct Node {
        int depth;
        bool isLeaf;
        Test test;
        Vote *vote;
        int leftChild; //nodeNumbers
        int rightChild;
        Node()
        {
            isLeaf = false;
            vote = NULL;
        }
        ~Node() {
            if(vote)
                delete vote;
        }
    };

    int maxDepth;
    int maxNodes;
    int nNodes;
    Node *nodes;

    //training samples
    int nSamples;
    Sample *samples;

    //training functions
    void TrainNode(int node, int nNodeSamples, int *nodeSamples)
    {
        int nTests;
        Test *tests;

        double bestSplitScore = BAD_SPLIT;

        int *leftSamples;
        int *rightSamples;
        int nLeftSamples;
        int nRightSamples;
        int bestNLeftSamples;
        int bestNRightSamples;

        nodes[node].isLeaf = false;


        if(nodes[node].depth < maxDepth)
        {
            DrawTests(nTests, &tests, nNodeSamples, nodeSamples);

            leftSamples = new int[nNodeSamples];
            rightSamples = new int[nNodeSamples];
            nLeftSamples;
            nRightSamples;

            bestNLeftSamples;
            bestNRightSamples;
            int bestTextIdx = 0;
            for(int i = 0; i < nTests; i++)
            {
                //spliting nodes
                nLeftSamples = 0;
                nRightSamples = 0;
                for(int j = 0; j < nNodeSamples; j++)
                {
                    if(tests[i].TestSample(&samples[nodeSamples[j]]))
                        rightSamples[nRightSamples++] = nodeSamples[j];
                    else
                        leftSamples[nLeftSamples++] = nodeSamples[j];
                }
                double splitScore = ScoreSplit(nLeftSamples, leftSamples, nRightSamples, rightSamples, tests+i);

                if(splitScore > bestSplitScore)
                {
                    bestSplitScore = splitScore;
                    nodes[node].test = tests[i];
                    bestTextIdx = i;
                    bestNLeftSamples = nLeftSamples;
                    bestNRightSamples = nRightSamples;

                    printf("bestSplitScore = %f\n", bestSplitScore);
                }

            }

            delete[] leftSamples;
            delete[] rightSamples;
            if(tests)
                delete[] tests;

        }

        if(bestSplitScore <= BAD_SPLIT+1)
        {
            nodes[node].isLeaf = true;
            nodes[node].vote = new Vote;
            AggregateVote(nNodeSamples,nodeSamples, nodes[node].vote);
            delete [] nodeSamples;
            return;
        }

        //recomputing the best split
        leftSamples = new int[bestNLeftSamples];
        rightSamples = new int[bestNRightSamples];
        nLeftSamples = 0;
        nRightSamples = 0;

        for(int j = 0; j < nNodeSamples; j++)
        {
            if(nodes[node].test.TestSample(&samples[nodeSamples[j]]))
                rightSamples[nRightSamples++] = nodeSamples[j];
            else
                leftSamples[nLeftSamples++] = nodeSamples[j];
        }
        assert(nLeftSamples == bestNLeftSamples && nRightSamples == bestNRightSamples);

        /*Vote vote, leftvote, rightvote;
    AggregateVote(nNodeSamples,nodeSamples, &vote);
    AggregateVote(nLeftSamples,leftSamples, &leftvote);
    AggregateVote(nRightSamples, rightSamples, &rightvote);*/

        delete[]  nodeSamples;

        //recurse to children
        nodes[node].leftChild = nNodes++;
        nodes[nNodes-1].depth = nodes[node].depth+1;
        TrainNode(nNodes-1, nLeftSamples, leftSamples);

        nodes[node].rightChild = nNodes++;
        nodes[nNodes-1].depth = nodes[node].depth+1;
        TrainNode(nNodes-1, nRightSamples, rightSamples);
    }

    //virtual functions needed to be redefined
    virtual void DrawSamples()  = 0;
    virtual void DrawTests(int& nTests, Test **tests, int nNodeSamples, int *nodeSamples)  = 0; //may return zero tests => the node will be made leaf

    virtual double ScoreSplit(int nLeftSamples, int *leftSamples, int nRightSamples, int *rightSamples, Test *test)  = 0; //may return BAD_SPLIT, if all splits are bad the node will be made leaf
    virtual void AggregateVote(int nNodeSamples, int *nodeSamples, Vote *vote)  = 0;


public:

    Vote *TestSample(Sample *s)
    {
        int node = 0;
        while(!nodes[node].isLeaf)
        {
            if(nodes[node].test.TestSample(s))
                node = nodes[node].rightChild;
            else
                node = nodes[node].leftChild;
        }
        //printf("%d\n", node);
        return nodes[node].vote;
    }

#ifdef VOTE_UPDATES
    void ResetVotes()
    {
        for(int i = 0; i < nNodes; i++)
            if(nodes[i].isLeaf)
                nodes[i].vote->ResetVote();
    }
    void UpdateVote(Sample *s)
    {
        int node = 0;
        while(!nodes[node].isLeaf)
        {
            if(nodes[node].test.TestSample(s))
                node = nodes[node].rightChild;
            else
                node = nodes[node].leftChild;
        }
        //printf("%d\n", node);
        nodes[node].vote->UpdateVote(s);
    }
#endif

    void Train(int max_depth)
    {
        maxDepth = max_depth;
        nodes = new Node[1 << maxDepth];
        if(!nodes)
        {
            std::cerr << "Not enough memory!" << std::endl;
            exit(0);
        }
        maxNodes = 1 << maxDepth;
        nNodes = 1;

        DrawSamples();
        nodes[0].depth = 1;

        int *nodeSamples = new int[nSamples];
        for(int i = 0; i < nSamples; i++)
            nodeSamples[i] = i;
        TrainNode(0, nSamples, nodeSamples);

    }

    void ReadFromFile(FILE *in, double blur_radius = 1)
    {
        fread((void *)&nNodes,sizeof(int), 1, in);
        nodes = new Node[nNodes];
        if(!nodes)
        {
            std::cerr << "Not enough memory!" << std::endl;
            exit(0);
        }

        for(int i = 0; i < nNodes; i++)
        {
            fread((void *)&nodes[i].depth,sizeof(int), 1, in);
            fread((void *)&nodes[i].isLeaf,sizeof(bool), 1, in);
            //printf("Node %d, leaf = %d, depth = %d\n", i, (int)nodes[i].isLeaf, (int)nodes[i].depth);
            if(nodes[i].isLeaf)
            {
                nodes[i].vote = new Vote;
                nodes[i].vote->ReadFromFile(in);
            }
            else
            {
                nodes[i].test.ReadFromFile(in);
                fread((void *)&nodes[i].rightChild,sizeof(int), 1, in);
                fread((void *)&nodes[i].leftChild,sizeof(int), 1, in);
                //printf("Left child = %d, right child = %d\n", nodes[i].rightChild, nodes[i].leftChild);
            }
        }
    }

    RandomTree() { nodes = NULL; samples = NULL; }
    ~RandomTree()
    {
        if(nodes) delete[] nodes; if(samples) delete[] samples;
    }

    void WriteToFile(FILE *out)
    {
        fwrite((void *)&nNodes,sizeof(int), 1, out);
        for(int i = 0; i < nNodes; i++)
        {
            fwrite((void *)&nodes[i].depth,sizeof(int), 1, out);
            fwrite((void *)&nodes[i].isLeaf,sizeof(bool), 1, out);

            //printf("Node %d, leaf = %d, depth = %d\n", i, (int)nodes[i].isLeaf, (int)nodes[i].depth);

            if(nodes[i].isLeaf)
                nodes[i].vote->WriteToFile(out);
            else
            {
                nodes[i].test.WriteToFile(out);
                fwrite((void *)&nodes[i].rightChild,sizeof(int), 1, out);
                fwrite((void *)&nodes[i].leftChild,sizeof(int), 1, out);
                //printf("Left child = %d, right child = %d\n", nodes[i].rightChild, nodes[i].leftChild);
            }
        }
    }
    /* DEBUG */
    void WriteToFileText(FILE *out)
    {
        fprintf(out, "nNodes = %d\n", nNodes);
        for(int i = 0; i < nNodes; i++)
        {
            fprintf(out, "nodes[%d].depth = %d\n", i, nodes[i].depth);
            int inIsLeaf = (nodes[i].isLeaf) ? 1 : 0;
            fprintf(out, "nodes[%d].isLeaf = %d\n", i, inIsLeaf);

            //printf("Node %d, leaf = %d, depth = %d\n", i, (int)nodes[i].isLeaf, (int)nodes[i].depth);

            if(nodes[i].isLeaf)
                nodes[i].vote->WriteToFileText(out);
            else
            {
                nodes[i].test.WriteToFileText(out);
                fprintf(out, "nodes[%d].rightChild = %d\n", i, nodes[i].rightChild);
                fprintf(out, "nodes[%d].leftChild = %d\n", i, nodes[i].leftChild);
                //printf("Left child = %d, right child = %d\n", nodes[i].rightChild, nodes[i].leftChild);
            }
        }
        return;
    }

}; // end of class RandomTree

