CFLAGS+=-g -Wall -pedantic
LDFLAGS+=-lm -lSDL2

.PHONY: all clean

all: gjkdemo

clean:
	rm -f gjkdemo

