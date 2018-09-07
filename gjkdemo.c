#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include <SDL2/SDL.h>

typedef struct {
	float x;
	float y;
} vector;

typedef struct {
	int count;
	vector *elems;
} vlist;

typedef struct {
	vlist vertices;
	vector position;
} Shape;

static vector v_neg(vector v)
{
	return (vector){-v.x, -v.y};
}

static float v_length(vector v)
{
	return sqrt(v.x * v.x + v.y * v.y);
}

static vector v_norm(vector v)
{
	float len = v_length(v);
	return (vector){v.x / len, v.y / len};
}

/* defined to be counter-clockwise. */
static vector v_perp(vector v)
{
	return (vector){-v.y, v.x};
}

/* defined to be counter-clockwise. */
static vector v_perp2(vector a, vector b)
{
	return (vector){a.y - b.y, b.x - a.x};
}

static vector v_add(vector lhs, vector rhs)
{
	return (vector){lhs.x + rhs.x, lhs.y + rhs.y};
}

static vector v_sub(vector lhs, vector rhs)
{
	return (vector){lhs.x - rhs.x, lhs.y - rhs.y};
}

static float v_dot(vector lhs, vector rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y;
}

static vlist vlist_new(int count, vector v[])
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

/* forward transform (from coordinate- to screen-space) */
static inline SDL_Point ftransform(vector v)
{
	return (SDL_Point){512 + v.x * 50, 384 - v.y * 50};
}

/* backward transform (from screen- to coordinate-space) */
static inline vector btransform(SDL_Point p)
{
	return (vector){(p.x - 512) / 50.0, (384 - p.y) / 50.0};
}

/* false -> counter-clockwise, true -> clockwise */
static bool triangle_winding(vector a, vector b, vector c)
{
	vector ab = v_sub(b, a), ac = v_sub(c, a);
	return (ab.x * ac.y - ab.y * ac.x) <= 0.0;
}

static vector best_vertex(vlist vertices, float (*score)(vector, void *), void *userdata)
{
	assert(vertices.count > 0);
	vector best_vertex = vertices.elems[0];
	float best_score = score(best_vertex, userdata);
	for (int i = 1; i < vertices.count; ++i) {
		vector cont_vertex = vertices.elems[i];
		float cont_score = score(cont_vertex, userdata);
		if (cont_score > best_score) {
			best_vertex = cont_vertex;
			best_score = cont_score;
		}
	}
	return best_vertex;
}

typedef struct {
	vector position;
	vector direction;
} support_polygon_ud;

static float support_polygon_score(vector v, void *userdata)
{
	support_polygon_ud ud = *(support_polygon_ud *)userdata;
	vector applied = v_add(v, ud.position);
	return v_dot(applied, ud.direction);
}

static vector support_polygon(Shape *shape, vector direction)
{
	support_polygon_ud ud = {shape->position, direction};
	return best_vertex(shape->vertices, support_polygon_score, &ud);
}

static int vis_depth;
static int vis_slice;
static int vis_span;
static Shape vis_shape;

static void vis_register_line(Shape *pair[2], vector a, vector b)
{
	if (++vis_depth != vis_slice) return;
	vector v[2] = {a, b};
	vlist l = vlist_new(2, v);
	vis_shape.position = (vector){0.0, 0.0};
	vis_shape.vertices = l;
}

static void vis_register_triangle(Shape *pair[2], vector a, vector b, vector c)
{
	if (++vis_depth != vis_slice) return;
	vector v[3] = {a, b, c};
	vlist l = vlist_new(3, v);
	vis_shape.position = (vector){0.0, 0.0};
	vis_shape.vertices = l;
}

static bool gjk_simplex3v(Shape *pair[2], vector dir, vector a, vector b);

static vector gjk_support(Shape *pair[2], vector dir)
{
	vector a = v_add(support_polygon(pair[0], dir), pair[0]->position);
	vector b = v_add(support_polygon(pair[1], v_neg(dir)), pair[1]->position);
	return v_sub(a, b);
}

static bool gjk_simplex2v(Shape *pair[2], vector dir, vector b)
{
	vector a = gjk_support(pair, dir);
	if (v_dot(a, dir) <= 0.0) return false;
	vis_register_line(pair, a, b);
	vector ab = v_sub(b, a), ao = v_neg(a);
	if (v_dot(ab, ao) >= 0.0) {
		vector n = v_perp(ab);
		if (v_dot(n, ao) > 0.0) {
			return gjk_simplex3v(pair, n, b, a);
		} else {
			return gjk_simplex3v(pair, v_neg(n), a, b);
		}
	} else {
		return gjk_simplex2v(pair, ao, a);
	}
}

static bool gjk_simplex3v(Shape *pair[2], vector dir, vector b, vector c)
{
	vector a = gjk_support(pair, dir);
	if (v_dot(a, dir) <= 0.0) return false;
	assert(triangle_winding(a, b, c) == true);
	vis_register_triangle(pair, a, b, c);
	vector ab = v_sub(b, a), ac = v_sub(c, a);
	vector nb = v_perp(ab), nc = v_perp(v_neg(ac));
	vector ao = v_neg(a);
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
	vector seed = gjk_support(pair, (vector){1.0, 0.0});
	return gjk_simplex2v(pair, v_neg(seed), seed);
}

static bool sgjk_test(Shape *pair[2], vector b, vector c);

static vector sgjk_support(Shape *pair[2], vector dir)
{
	vector a = v_add(support_polygon(pair[0], dir), pair[0]->position);
	vector b = v_add(support_polygon(pair[1], v_neg(dir)), pair[1]->position);
	return v_sub(a, b);
}

static bool sgjk_crawl(Shape *pair[2], vector p, vector a, vector b, vector c)
{
	return v_dot(p, a) > 0.0 ? sgjk_test(pair, a, c) : sgjk_test(pair, b, a);
}

static bool sgjk_test(Shape *pair[2], vector b, vector c)
{
	vector dir = v_perp2(c, b), a = sgjk_support(pair, dir);
	if (v_dot(a, dir) <= 0.0) return false;
	assert(triangle_winding(a, b, c) == true);
	vis_register_triangle(pair, a, b, c);
	vector pb = v_perp2(a, b), pc = v_perp2(c, a);
	if (v_dot(pb, a) >= 0.0 && v_dot(pc, a) >= 0.0) return true;
	return sgjk_crawl(pair, v_add(v_norm(pb), v_neg(v_norm(pc))), a, b, c);
}

static bool sgjk(Shape *pair[2])
{
	vector a = sgjk_support(pair, (vector){ 1.0, 0.0});
	vector b = sgjk_support(pair, (vector){-1.0, 0.0});
	return sgjk_crawl(pair, v_perp2(a, b), a, b, b);
}

typedef struct {
	vector base;
	vector direction;
} vscore_project_ud;

static float vscore_project(vector v, void *userdata)
{
	vscore_project_ud ud = *(vscore_project_ud *)userdata;
	float score = v_dot(v_sub(v, ud.base), ud.direction);
	return score;
}

static float vscore_minx(vector v, void *userdata)
{
	(void) userdata;
	return -v.x;
}

static float vscore_maxx(vector v, void *userdata)
{
	(void) userdata;
	return v.x;
}

static vlist mink_quickhull_r(vlist cloud, vector a, vector b)
{
	vector n = v_norm(v_perp(v_sub(b, a)));
	vscore_project_ud ud = {.base = a, .direction = n};
	vector p = best_vertex(cloud, vscore_project, &ud);
	float distance = vscore_project(p, &ud);
	if (distance > 0.0001) {
		return vlist_join(mink_quickhull_r(cloud, a, p), mink_quickhull_r(cloud, p, b));
	} else {
		return vlist_new(1, &b);
	}
}

static Shape mink_quickhull(vlist cloud, vector position)
{
	vector min = best_vertex(cloud, vscore_minx, NULL);
	vector max = best_vertex(cloud, vscore_maxx, NULL);
	vlist hull = vlist_new(1, &min);
	hull = vlist_join(hull, mink_quickhull_r(cloud, min, max));
	hull = vlist_join(hull, mink_quickhull_r(cloud, max, min));
	return (Shape){hull, position};
}

static vlist mink_cloud(vlist a, vlist b)
{
	vlist cloud = {a.count * b.count};
	cloud.elems = calloc(cloud.count, sizeof(*cloud.elems));
	for (int i = 0; i < a.count; ++i) {
		for (int j = 0; j < b.count; ++j) {
			cloud.elems[i * b.count + j] = v_sub(a.elems[i], b.elems[j]);
		}
	}
	return cloud;
}

static Shape mink(Shape a, Shape b)
{
	vlist cloud = mink_cloud(a.vertices, b.vertices);
	Shape hull = mink_quickhull(cloud, v_sub(a.position, b.position));
	free(cloud.elems);
	return hull;
}

static SDL_Window *demo_window;
static SDL_Renderer *demo_renderer;

static vector demo_shape1_buf[4] = {
	{-1, -1},
	{-1,  1},
	{ 1,  1},
	{ 1, -1},
};

static vector demo_shape2_buf[3] = {
	{ 0, -1},
	{-1,  1},
	{ 1,  1},
};

static Shape demo_shape1 = {{4, demo_shape1_buf}, {0.0, 0.0}};
static Shape demo_shape2 = {{3, demo_shape2_buf}, {3.0, 1.0}};
static float demo_slider = 0.75;
typedef enum { GJK, SGJK } demo_Variant;
static demo_Variant demo_variant;
static bool demo_is_running = true;
static enum {
	Grab_None,
	Grab_Shape1,
	Grab_Shape2,
	Grab_Slider
} demo_grabbed;
static SDL_Point demo_mouse_prev;
static bool demo_colliding;
static Shape demo_mink_shape;

static bool demo_detect_collision(Shape *s1, Shape *s2)
{
	vis_depth = 0;
	free(vis_shape.vertices.elems);
	vis_shape.vertices = (vlist){0, NULL};
	Shape *pair[2] = {s1, s2};
	bool r;
	switch (demo_variant) {
	case GJK:
		r = gjk(pair);
		break;
	case SGJK:
		r = sgjk(pair);
		break;
	}
	vis_span = vis_depth;
	return r;
}

static void demo_select_variant(demo_Variant selection)
{
	demo_variant = selection;
	switch (selection) {
	case GJK:
		SDL_SetWindowTitle(demo_window, "Currently using: GJK  --  2D GJK Demo  --  (Thomas Oltmann)");
		break;
	case SGJK:
		SDL_SetWindowTitle(demo_window, "Currently using: Simplified GJK  --  2D GJK Demo  --  (Thomas Oltmann)");
		break;
	}
}

static void demo_handle_events(void)
{
	SDL_Event event;
	SDL_Point cur_p;
	vector cur_v, rel_v;
	Shape cursor;
	while (SDL_PollEvent(&event)) {
		switch (event.type) {
		case SDL_KEYUP:
			if (event.key.keysym.sym == SDLK_TAB) {
				demo_select_variant((demo_variant + 1) % 2);
			}
			break;
		case SDL_MOUSEBUTTONDOWN:
			if (demo_grabbed != Grab_None) break;
			cur_p = (SDL_Point){event.button.x, event.button.y};
			cur_v = btransform(cur_p);
			cursor = (Shape){{1, &cur_v}, {0.0, 0.0}};
			demo_mouse_prev = cur_p;
			if (demo_detect_collision(&cursor, &demo_shape1)) {
				demo_grabbed = Grab_Shape1;
			} else if (demo_detect_collision(&cursor, &demo_shape2)) {
				demo_grabbed = Grab_Shape2;
			} else if (cur_p.y >= 768 - 32) {
				demo_grabbed = Grab_Slider;
			} else {
				demo_grabbed = Grab_None;
			}
			break;
		case SDL_MOUSEBUTTONUP:
			demo_grabbed = Grab_None;
			break;
		case SDL_MOUSEMOTION:
			cur_p = (SDL_Point){event.motion.x, event.motion.y};
			rel_v = v_sub(btransform(cur_p), btransform(demo_mouse_prev));
			demo_mouse_prev = cur_p;
			switch (demo_grabbed) {
			case Grab_Shape1:
				demo_shape1.position = v_add(demo_shape1.position, rel_v);
				break;
			case Grab_Shape2:
				demo_shape2.position = v_add(demo_shape2.position, rel_v);
				break;
			case Grab_Slider:
				demo_slider = event.motion.x / 1024.0;
				break;
			case Grab_None: break;
			}
			break;
		case SDL_QUIT:
			SDL_Log("Quit after %i ticks.", event.quit.timestamp);
			demo_is_running = false;
			break;
		}
	}
}

static void demo_draw_origin(void)
{
	SDL_Point o = {512, 384};
	SDL_SetRenderDrawColor(demo_renderer, 100, 100, 100, SDL_ALPHA_OPAQUE);
	SDL_RenderDrawLine(demo_renderer, o.x - 512, o.y, o.x + 512, o.y);
	SDL_RenderDrawLine(demo_renderer, o.x, o.y - 384, o.x, o.y + 384);
}

static void demo_draw_shape(Shape *shape)
{
	SDL_Point points[shape->vertices.count + 1];
	for (int i = 0; i < shape->vertices.count; ++i) {
		points[i] = ftransform(v_add(shape->vertices.elems[i], shape->position));
	}
	points[shape->vertices.count] = points[0];
	SDL_RenderDrawLines(demo_renderer, points, shape->vertices.count + 1);
}

static void demo_draw_slider(void)
{
	SDL_SetRenderDrawColor(demo_renderer, 0, 0, 255, SDL_ALPHA_OPAQUE);
	SDL_Rect slider_rect = {0, 768 - 32, demo_slider * 1024, 32};
	SDL_RenderFillRect(demo_renderer, &slider_rect);
}

static void demo_draw(void)
{
	SDL_SetRenderDrawColor(demo_renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
	SDL_RenderClear(demo_renderer);
	demo_draw_origin();
	SDL_SetRenderDrawColor(demo_renderer, 0, 255, 0, SDL_ALPHA_OPAQUE);
	demo_draw_shape(&demo_mink_shape);
	if (demo_colliding) {
		SDL_SetRenderDrawColor(demo_renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
	} else {
		SDL_SetRenderDrawColor(demo_renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
	}
	demo_draw_shape(&demo_shape1);
	demo_draw_shape(&demo_shape2);
	SDL_SetRenderDrawColor(demo_renderer, 0, 0, 255, SDL_ALPHA_OPAQUE);
	demo_draw_shape(&vis_shape);
	demo_draw_slider();
	SDL_RenderPresent(demo_renderer);
}

int main(int argc, char *argv[])
{
	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
	demo_window = SDL_CreateWindow("",
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		1024, 768, 0);
	demo_renderer = SDL_CreateRenderer(demo_window, -1, SDL_RENDERER_ACCELERATED);
	demo_select_variant(GJK);
	while (demo_is_running) {
		demo_handle_events();
		vis_slice = roundf(demo_slider * vis_span);
		demo_mink_shape = mink(demo_shape1, demo_shape2);
		demo_colliding = demo_detect_collision(&demo_shape1, &demo_shape2);
		demo_draw();
		free(demo_mink_shape.vertices.elems);
	}
	SDL_DestroyRenderer(demo_renderer);
	SDL_DestroyWindow(demo_window);
	SDL_Quit();
	return 0;
}
