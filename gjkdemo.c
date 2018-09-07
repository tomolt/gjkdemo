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
static Shape shape2 = {{3, shape2_buf}, {3.0, 1.0}};

int gjk_depth;
int vis_slice;
int gjk_span;
float slider = 0.75;
Shape vis_shape;
enum { GJK, SGJK } variant;

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

/* defined to be counter-clockwise. */
static vertex v_perp2(vertex a, vertex b)
{
	return (vertex){a.y - b.y, b.x - a.x};
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

static float v_distance(vertex lhs, vertex rhs)
{
	return v_length(v_sub(rhs, lhs));
}

/* forward transform (from coordinate- to screen-space) */
static inline SDL_Point ftransform(vertex v)
{
	return (SDL_Point){512 + v.x * 50, 384 - v.y * 50};
}

/* backward transform (from screen- to coordinate-space) */
static inline vertex btransform(SDL_Point p)
{
	return (vertex){(p.x - 512) / 50.0, (-p.y + 384) / 50.0};
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

static vlist vlist_new(int count, vertex v[])
{
	vlist l = {count};
	l.elems = calloc(count, sizeof(*l.elems));
	memcpy(l.elems, v, count * sizeof(*l.elems));
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

/* false -> counter-clockwise, true -> clockwise */
static bool triangle_winding(vertex a, vertex b, vertex c)
{
	vertex ab = v_sub(b, a), ac = v_sub(c, a);
	return (ab.x * ac.y - ab.y * ac.x) <= 0.0;
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

static void gjk_visualize_line(Shape *pair[2], vertex a, vertex b)
{
	if (++gjk_depth != vis_slice) return;
	vertex v[2] = {a, b};
	vlist l = vlist_new(2, v);
	vis_shape.position = (vertex){0.0, 0.0};
	vis_shape.vertices = l;
}

static void gjk_visualize_triangle(Shape *pair[2], vertex a, vertex b, vertex c)
{
	if (++gjk_depth != vis_slice) return;
	vertex v[3] = {a, b, c};
	vlist l = vlist_new(3, v);
	vis_shape.position = (vertex){0.0, 0.0};
	vis_shape.vertices = l;
}

static bool gjk_simplex3v(Shape *pair[2], vertex dir, vertex a, vertex b);

static vertex gjk_support(Shape *pair[2], vertex dir)
{
	vertex a = v_add(support_polygon(pair[0], dir), pair[0]->position);
	vertex b = v_add(support_polygon(pair[1], v_neg(dir)), pair[1]->position);
	return v_sub(a, b);
}

static bool gjk_simplex2v(Shape *pair[2], vertex dir, vertex b)
{
	vertex a = gjk_support(pair, dir);
	if (v_dot(a, dir) <= 0.0) return false;
	gjk_visualize_line(pair, a, b);
	vertex ab = v_sub(b, a), ao = v_neg(a);
	if (v_dot(ab, ao) >= 0.0) {
		vertex n = v_perp(ab);
		if (v_dot(n, ao) > 0.0) {
			return gjk_simplex3v(pair, n, b, a);
		} else {
			return gjk_simplex3v(pair, v_neg(n), a, b);
		}
	} else {
		return gjk_simplex2v(pair, ao, a);
	}
}

static bool gjk_simplex3v(Shape *pair[2], vertex dir, vertex b, vertex c)
{
	vertex a = gjk_support(pair, dir);
	if (v_dot(a, dir) <= 0.0) return false;
	assert(triangle_winding(a, b, c) == true);
	gjk_visualize_triangle(pair, a, b, c);
	vertex ab = v_sub(b, a), ac = v_sub(c, a);
	vertex nb = v_perp(ab), nc = v_perp(v_neg(ac));
	vertex ao = v_neg(a);
	if (v_dot(nb, ao) > 0.0) {
		if (v_dot(ab, ao) > 0.0) {
			return gjk_simplex3v(pair, nb, b, a);
		} else {
			return gjk_simplex2v(pair, ao, a);
		}
	} else if (v_dot(nc, ao) > 0.0) {
		if (v_dot(ab, ao) > 0.0) {
			return gjk_simplex3v(pair, nc, a, c);
		} else {
			return gjk_simplex2v(pair, ao, a);
		}
	} else {
		return true;
	}
}

static bool gjk(Shape *pair[2])
{
	vertex seed = gjk_support(pair, (vertex){1.0, 0.0});
	return gjk_simplex2v(pair, v_neg(seed), seed);
}

static bool sgjk_test(Shape *pair[2], vertex b, vertex c);

static vertex sgjk_support(Shape *pair[2], vertex dir)
{
	vertex a = v_add(support_polygon(pair[0], dir), pair[0]->position);
	vertex b = v_add(support_polygon(pair[1], v_neg(dir)), pair[1]->position);
	return v_sub(a, b);
}

static bool sgjk_crawl(Shape *pair[2], vertex p, vertex a, vertex b, vertex c)
{
	return v_dot(p, a) > 0.0 ? sgjk_test(pair, a, c) : sgjk_test(pair, b, a);
}

static bool sgjk_test(Shape *pair[2], vertex b, vertex c)
{
	vertex dir = v_perp2(c, b), a = sgjk_support(pair, dir);
	if (v_dot(a, dir) <= 0.0) return false;
	assert(triangle_winding(a, b, c) == true);
	gjk_visualize_triangle(pair, a, b, c);
	vertex pb = v_perp2(a, b), pc = v_perp2(c, a);
	if (v_dot(pb, a) >= 0.0 && v_dot(pc, a) >= 0.0) return true;
	return sgjk_crawl(pair, v_add(v_norm(pb), v_neg(v_norm(pc))), a, b, c);
}

static bool sgjk(Shape *pair[2])
{
	vertex a = sgjk_support(pair, (vertex){ 1.0, 0.0});
	vertex b = sgjk_support(pair, (vertex){-1.0, 0.0});
	return sgjk_crawl(pair, v_perp2(a, b), a, b, b);
}

static bool detect_collision(Shape *s1, Shape *s2)
{
	gjk_depth = 0;
	free(vis_shape.vertices.elems);
	vis_shape.vertices = (vlist){0, NULL};
	Shape *pair[2] = {s1, s2};
	bool r;
	switch (variant) {
	case GJK:
		r = gjk(pair);
		break;
	case SGJK:
		r = sgjk(pair);
		break;
	}
	gjk_span = gjk_depth;
	return r;
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
		return vlist_new(1, &b);
	}
}

static Shape construct_hull(Shape shape)
{
	vertex min = best_vertex(shape.vertices, vscore_minx, NULL);
	vertex max = best_vertex(shape.vertices, vscore_maxx, NULL);
	vlist vertices = vlist_new(1, &min);
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
	SDL_SetRenderDrawColor(renderer, 100, 100, 100, SDL_ALPHA_OPAQUE);
	SDL_RenderDrawLine(renderer, o.x - 512, o.y, o.x + 512, o.y);
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

typedef enum {
	Grab_None,
	Grab_Shape1,
	Grab_Shape2,
	Grab_Slider
} Grabbable;

Grabbable grabbed;

int main(int argc, char *argv[])
{
	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
	window = SDL_CreateWindow("gjkdemo",
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		1024, 768, 0);
	renderer = SDL_CreateRenderer(window, -1,
		SDL_RENDERER_ACCELERATED);
	bool running = true;
	while (running) {
		SDL_Event event;
		SDL_PollEvent(&event);
		switch (event.type) {
		case SDL_KEYUP:
			if (event.key.keysym.sym == SDLK_SPACE) {
				debug_hook();
			} else if (event.key.keysym.sym == SDLK_TAB) {
				variant = (variant + 1) % 2;
				switch (variant) {
				case GJK:
					SDL_SetWindowTitle(window, "2D GJK Demo   (Thomas Oltmann)");
					break;
				case SGJK:
					SDL_SetWindowTitle(window, "2D SGJK Demo   (Thomas Oltmann)");
					break;
				}
			}
			break;
		case SDL_MOUSEBUTTONDOWN: {
			SDL_Point mp = {event.button.x, event.button.y};
			vertex mv = btransform(mp);
			if (grabbed == Grab_None) {
				if (v_distance(mv, shape1.position) <= 0.5) {
					grabbed = Grab_Shape1;
				} else if (v_distance(mv, shape2.position) <= 0.5) {
					grabbed = Grab_Shape2;
				} else if (mp.y >= 768 - 32) {
					grabbed = Grab_Slider;
				} else {
					grabbed = Grab_None;
				}
			}
			} break;
		case SDL_MOUSEBUTTONUP:
			grabbed = Grab_None;
			break;
		case SDL_MOUSEMOTION: {
			SDL_Point mp = {event.motion.x, event.motion.y};
			vertex mv = btransform(mp);
			switch (grabbed) {
			case Grab_Shape1:
				shape1.position = mv;
				break;
			case Grab_Shape2:
				shape2.position = mv;
				break;
			case Grab_Slider:
				slider = mp.x / 1024.0;
				break;
			case Grab_None: break;
			}
			} break;
		case SDL_QUIT:
			SDL_Log("Quit after %i ticks.", event.quit.timestamp);
			running = false;
			break;
		}
		vis_slice = roundf(slider * gjk_span);
		Shape dshape = construct_hull(difference_shape(shape1, shape2));
		bool colliding = detect_collision(&shape1, &shape2);
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
		SDL_RenderClear(renderer);
		render_origin();
		SDL_SetRenderDrawColor(renderer, 0, 255, 0, SDL_ALPHA_OPAQUE);
		render_shape(&dshape);
		if (colliding) {
			SDL_SetRenderDrawColor(renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
		} else {
			SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
		}
		render_shape(&shape1);
		render_shape(&shape2);
		SDL_SetRenderDrawColor(renderer, 0, 0, 255, SDL_ALPHA_OPAQUE);
		render_shape(&vis_shape);
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
	}
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
