#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include <SDL2/SDL.h>

typedef struct {
	float x;
	float y;
} vertex;

typedef struct {
	int count;
	vertex *elems;
} vlist;

typedef struct {
	vlist vertices;
	vertex position;
} Shape;

static SDL_Window *window;
static SDL_Renderer *renderer;

static vertex shape1_buf[4] = {
	{-1, -1},
	{-1,  1},
	{ 1,  1},
	{ 1, -1},
};

static vertex shape2_buf[3] = {
	{ 0, -1},
	{-1,  1},
	{ 1,  1},
};

static Shape shape1 = {{4, shape1_buf}, {0.0, 0.0}};
static Shape shape2 = {{3, shape2_buf}, {0.0, 0.0}};

float slider;

static vertex v_neg(vertex v)
{
	return (vertex){-v.x, -v.y};
}

static float v_length(vertex v)
{
	return sqrt(v.x * v.x + v.y * v.y);
}

static vertex v_norm(vertex v)
{
	float len = v_length(v);
	return (vertex){v.x / len, v.y / len};
}

/* defined to be counter-clockwise. */
static vertex v_perp(vertex v)
{
	return (vertex){-v.y, v.x};
}

static vertex v_add(vertex lhs, vertex rhs)
{
	return (vertex){lhs.x + rhs.x, lhs.y + rhs.y};
}

static vertex v_sub(vertex lhs, vertex rhs)
{
	return (vertex){lhs.x - rhs.x, lhs.y - rhs.y};
}

static float v_dot(vertex lhs, vertex rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y;
}

/* TODO better name */
static vertex v_perp2(vertex from, vertex to)
{
	return v_perp(v_sub(to, from));
}

static vertex best_vertex(vlist vertices, float (*score)(vertex, void *), void *userdata)
{
	assert(vertices.count > 0);
	vertex best_vertex = vertices.elems[0];
	float best_score = score(best_vertex, userdata);
	for (int i = 1; i < vertices.count; ++i) {
		vertex cont_vertex = vertices.elems[i];
		float cont_score = score(cont_vertex, userdata);
		if (cont_score > best_score) {
			best_vertex = cont_vertex;
			best_score = cont_score;
		}
	}
	return best_vertex;
}

static vlist vlist_new(vertex v)
{
	vlist l = {1};
	l.elems = calloc(l.count, sizeof(*l.elems));
	l.elems[0] = v;
	return l;
}

static vlist vlist_join(vlist a, vlist b)
{
	vlist l;
	l.count = a.count + b.count;
	l.elems = calloc(l.count, sizeof(*l.elems));
	memcpy(l.elems          , a.elems, a.count * sizeof(*l.elems));
	memcpy(l.elems + a.count, b.elems, b.count * sizeof(*l.elems));
	free(a.elems);
	free(b.elems);
	return l;
}

typedef struct {
	vertex position;
	vertex direction;
} support_polygon_ud;

static float support_polygon_score(vertex v, void *userdata)
{
	support_polygon_ud ud = *(support_polygon_ud *)userdata;
	vertex applied = v_add(v, ud.position);
	return v_dot(applied, ud.direction);
}

static vertex support_polygon(Shape *shape, vertex direction)
{
	support_polygon_ud ud = {shape->position, direction};
	return best_vertex(shape->vertices, support_polygon_score, &ud);
}

static vertex minkowski_support(Shape *pair[2], vertex dir)
{
	vertex a = support_polygon(pair[0], dir);
	vertex b = support_polygon(pair[1], dir);
	return (vertex){a.x - b.x, a.y - b.y};
}

static bool gjk_simplex3d(Shape *pair[2], vertex dir, vertex a, vertex b);

static bool gjk_simplex2d(Shape *pair[2], vertex dir, vertex b)
{
	vertex a = minkowski_support(pair, dir);
	if (v_dot(a, dir) < 0.0) return false;
	vertex ab = v_sub(b, a), ao = v_neg(a);
	if (v_dot(ab, ao) > 0.0) {
		vertex n = v_perp(ab);
		if (v_dot(n, ao) > 0.0) {
			return gjk_simplex3d(pair, n, b, a);
		} else {
			return gjk_simplex3d(pair, v_neg(n), a, b);
		}
	} else {
		return gjk_simplex2d(pair, ao, a);
	}
}

static bool gjk_simplex3d(Shape *pair[2], vertex dir, vertex b, vertex c)
{
	vertex a = minkowski_support(pair, dir);
	if (v_dot(a, dir) < 0.0) return false;
	vertex ab = v_sub(b, a), ac = v_sub(c, a);
	vertex nb = v_perp(ab), nc = v_perp(v_neg(ac));
	vertex ao = v_neg(a);
	if (v_dot(nb, ao) < 0.0 && v_dot(nc, ao) < 0.0) {
		return true;
	} else if (v_dot(ab, ao) > 0.0) {
		return gjk_simplex3d(pair, nb, b, a);
	} else if (v_dot(ac, ao) > 0.0) {
		return gjk_simplex3d(pair, nc, c, a);
	} else {
		return gjk_simplex2d(pair, ao, a);
	}
}

static bool detect_collision(Shape *s1, Shape *s2)
{
	Shape *pair[2] = {s1, s2};
	vertex init_dir = v_sub(s1->position, s2->position);
	vertex seed = minkowski_support(pair, init_dir);
	return gjk_simplex2d(pair, v_neg(seed), seed);
}

typedef struct {
	vertex base;
	vertex direction;
} vscore_project_ud;

static float vscore_project(vertex v, void *userdata)
{
	vscore_project_ud ud = *(vscore_project_ud *)userdata;
	float score = v_dot(v_sub(v, ud.base), ud.direction);
	return score;
}

static float vscore_minx(vertex v, void *userdata)
{
	(void) userdata;
	return -v.x;
}

static float vscore_maxx(vertex v, void *userdata)
{
	(void) userdata;
	return v.x;
}

static vlist qhull_fractal(vlist cloud, vertex a, vertex b)
{
	vertex n = v_norm(v_perp(v_sub(b, a)));
	vscore_project_ud ud = {.base = a, .direction = n};
	vertex p = best_vertex(cloud, vscore_project, &ud);
	float distance = vscore_project(p, &ud);
	if (distance > 0.0001) {
		return vlist_join(qhull_fractal(cloud, a, p), qhull_fractal(cloud, p, b));
	} else {
		return vlist_new(b);
	}
}

static Shape construct_hull(Shape shape)
{
	vertex min = best_vertex(shape.vertices, vscore_minx, NULL);
	vertex max = best_vertex(shape.vertices, vscore_maxx, NULL);
	vlist vertices = vlist_new(min);
	vertices = vlist_join(vertices, qhull_fractal(shape.vertices, min, max));
	vertices = vlist_join(vertices, qhull_fractal(shape.vertices, max, min));
	free(shape.vertices.elems);
	return (Shape){vertices, shape.position};
}

static vlist difference_vlist(vlist a, vlist b)
{
	vlist diff = {a.count * b.count};
	diff.elems = calloc(diff.count, sizeof(*diff.elems));
	for (int i = 0; i < a.count; ++i) {
		for (int j = 0; j < b.count; ++j) {
			diff.elems[i * b.count + j] = v_sub(a.elems[i], b.elems[j]);
		}
	}
	return diff;
}

static Shape difference_shape(Shape a, Shape b)
{
	return (Shape){
		.vertices = difference_vlist(a.vertices, b.vertices),
		.position = v_sub(a.position, b.position)
	};
}

static void render_origin(void)
{
	SDL_Point o = {512, 384};
	SDL_SetRenderDrawColor(renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
	SDL_RenderDrawLine(renderer, o.x - 512, o.y, o.x + 512, o.y);
	SDL_SetRenderDrawColor(renderer, 0, 0, 255, SDL_ALPHA_OPAQUE);
	SDL_RenderDrawLine(renderer, o.x, o.y - 384, o.x, o.y + 384);
}

static void render_shape(Shape *shape)
{
	SDL_Point points[shape->vertices.count + 1];
	for (int i = 0; i < shape->vertices.count; ++i) {
		points[i] = (SDL_Point){
			512 + (shape->vertices.elems[i].x + shape->position.x) * 50,
			384 - (shape->vertices.elems[i].y + shape->position.y) * 50,
		};
	}
	points[shape->vertices.count] = points[0];
	SDL_RenderDrawLines(renderer, points, shape->vertices.count + 1);
}

static void debug_hook(void) {}

int main(int argc, char *argv[])
{
	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
	window = SDL_CreateWindow("2D GJK Demo   (Thomas Oltmann)",
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		1024, 768, 0);
	renderer = SDL_CreateRenderer(window, -1,
		SDL_RENDERER_ACCELERATED);
	bool running = true;
	unsigned long long tmp = 0;
	while (running) {
		SDL_Event event;
		SDL_PollEvent(&event);
		switch (event.type) {
		case SDL_KEYUP:
			if (event.key.keysym.sym == SDLK_SPACE) {
				debug_hook();
			}
			break;
		case SDL_MOUSEMOTION:
			shape2.position = (vertex){
				(event.motion.x - 512) / 50.0,
				(-event.motion.y + 384) / 50.0
			};
			break;
		case SDL_QUIT:
			SDL_Log("Quit after %i ticks.", event.quit.timestamp);
			running = false;
			break;
		}
		slider = (sin(tmp / 10000.0) + 1.0) / 2.0;
		Shape dshape = construct_hull(difference_shape(shape1, shape2));
		// bool colliding = detect_collision(&shape1, &shape2);
		bool colliding = false;
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
		SDL_RenderClear(renderer);
		render_origin();
		if (colliding) {
			SDL_SetRenderDrawColor(renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
		} else {
			SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
		}
		render_shape(&shape1);
		render_shape(&shape2);
		SDL_SetRenderDrawColor(renderer, 0, 255, 0, SDL_ALPHA_OPAQUE);
		render_shape(&dshape);
		SDL_SetRenderDrawColor(renderer, 0, 0, 255, SDL_ALPHA_OPAQUE);
		SDL_Rect slider_rect = {
			0,
			768 - 32,
			slider * 1024,
			32
		};
		SDL_RenderFillRect(renderer, &slider_rect);
		SDL_RenderPresent(renderer);
		free(dshape.vertices.elems);
		++tmp;
	}
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
