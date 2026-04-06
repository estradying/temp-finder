CC = gcc
CFLAGS = -O3 -march=native -ffast-math -funroll-loops -flto -fomit-frame-pointer -fstrict-aliasing -pthread
LDFLAGS = -lm -pthread

TARGET = main
SRC = main.c \
      cubiomes/biomenoise.c \
      cubiomes/biomes.c \
      cubiomes/finders.c \
      cubiomes/generator.c \
      cubiomes/noise.c \
      cubiomes/quadbase.c \
      cubiomes/util.c

all:
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)