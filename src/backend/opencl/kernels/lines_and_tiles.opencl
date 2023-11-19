R"(

struct Line2
{
	float x0, x1;
};
struct Line3
{
	float x0, x1, x2;
};
struct Line4
{
	float x0, x1, x2, x3;
};
struct Line5
{
	float x0, x1, x2, x3, x4;
};
struct Line6
{
	float x0, x1, x2, x3, x4, x5;
};

float get_line2_at(struct Line2 line, int index)
{
	return ((float*)(&line))[index];
}
float get_line3_at(struct Line3 line, int index)
{
	return ((float*)(&line))[index];
}
float get_line4_at(struct Line4 line, int index)
{
	return ((float*)(&line))[index];
}
float get_line5_at(struct Line5 line, int index)
{
	return ((float*)(&line))[index];
}
float get_line6_at(struct Line6 line, int index)
{
	return ((float*)(&line))[index];
}

void set_line2_at(__private struct Line2 *line, int index, float value)
{
	((float*)(line))[index] = value;
}
void set_line3_at(__private struct Line3 *line, int index, float value)
{
	((float*)(line))[index] = value;
}
void set_line4_at(__private struct Line4 *line, int index, float value)
{
	((float*)(line))[index] = value;
}
void set_line5_at(__private struct Line5 *line, int index, float value)
{
	((float*)(line))[index] = value;
}
void set_line6_at(__private struct Line6 *line, int index, float value)
{
	((float*)(line))[index] = value;
}

struct Tile2x2
{
	struct Line2 x0, x1;
};
struct Tile3x3
{
	struct Line3 x0, x1, x2;
};
struct Tile4x4
{
	struct Line4 x0, x1, x2, x3;
};
struct Tile5x5
{
	struct Line5 x0, x1, x2, x3, x4;
};
struct Tile6x6
{
	struct Line6 x0, x1, x2, x3, x4, x5;
};

struct Line2 get_row_from_tile2x2(struct Tile2x2 tile, int row)
{
	return ((struct Line2*)(&tile))[row];
}
struct Line3 get_row_from_tile3x3(struct Tile3x3 tile, int row)
{
	return ((struct Line3*)(&tile))[row];
}
struct Line4 get_row_from_tile4x4(struct Tile4x4 tile, int row)
{
	return ((struct Line4*)(&tile))[row];
}
struct Line5 get_row_from_tile5x5(struct Tile5x5 tile, int row)
{
	return ((struct Line5*)(&tile))[row];
}
struct Line6 get_row_from_tile6x6(struct Tile6x6 tile, int row)
{
	return ((struct Line6*)(&tile))[row];
}

float get_tile2x2_at(struct Tile2x2 tile, int row, int col)
{
	return ((float*)(&tile))[row * 2 + col];
}
float get_tile3x3_at(struct Tile3x3 tile, int row, int col)
{
	return ((float*)(&tile))[row * 3 + col];
}
float get_tile4x4_at(struct Tile4x4 tile, int row, int col)
{
	return ((float*)(&tile))[row * 4 + col];
}
float get_tile5x5_at(struct Tile5x5 tile, int row, int col)
{
	return ((float*)(&tile))[row * 5 + col];
}
float get_tile6x6_at(struct Tile6x6 tile, int row, int col)
{
	return ((float*)(&tile))[row * 6 + col];
}

void set_tile2x2_at(__private struct Tile2x2 *tile, int row, int col, float value)
{
	((float*)(tile))[row * 2 + col] = value;
}
void set_tile3x3_at(__private struct Tile3x3 *tile, int row, int col, float value)
{
	((float*)(tile))[row * 3 + col] = value;
}
void set_tile4x4_at(__private struct Tile4x4 *tile, int row, int col, float value)
{
	((float*)(tile))[row * 4 + col] = value;
}
void set_tile5x5_at(__private struct Tile5x5 *tile, int row, int col, float value)
{
	((float*)(tile))[row * 5 + col] = value;
}
void set_tile6x6_at(__private struct Tile6x6 *tile, int row, int col, float value)
{
	((float*)(tile))[row * 6 + col] = value;
}

)"