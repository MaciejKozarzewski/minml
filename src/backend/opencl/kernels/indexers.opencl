R"(

/*
* Indexer 1D
*/
struct Indexer1D
{
	int length;
};
struct Indexer1D create_indexer_1D(int dim0)
{
	struct Indexer1D result;
	result.length = dim0;
	return result;
}
int get_pos_1D(struct Indexer1D ind, int x0)
{
	return x0;
}

/*
* Indexer 2D
*/
struct Indexer2D
{
	int stride0;
};
struct Indexer2D create_indexer_2D(int dim0, int dim1)
{
	struct Indexer2D result;
	result.stride0 = dim1;
	return result;
}
int get_pos_2D(struct Indexer2D ind, int x0, int x1)
{
	return x0 * ind.stride0 + x1;
}

/*
* Indexer 3D
*/
struct Indexer3D
{
	int stride0, stride1;
};
struct Indexer3D create_indexer_3D(int dim0, int dim1, int dim2)
{
	struct Indexer3D result;
	result.stride0 = dim1 * dim2;
	result.stride1 = dim2;
	return result;
}
int get_pos_3D(struct Indexer3D ind, int x0, int x1, int x2)
{
	return x0 * ind.stride0 + x1 * ind.stride1 + x2;
}

/*
* Indexer 4D
*/
struct Indexer4D
{
	int stride0, stride1, stride2;
};
struct Indexer4D create_indexer_4D(int dim0, int dim1, int dim2, int dim3)
{
	struct Indexer4D result;
	result.stride0 = dim1 * dim2 * dim3;
	result.stride1 = dim2 * dim3;
	result.stride2 = dim3;
	return result;
}
int get_pos_4D(struct Indexer4D ind, int x0, int x1, int x2, int x3)
{
	return x0 * ind.stride0 + x1 * ind.stride1 + x2 * ind.stride2 + x3;
}

)"