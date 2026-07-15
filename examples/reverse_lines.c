#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void reverse(char *begin, char *end) {
    while (begin < end) {
        char tmp = *begin;
        *begin++ = *end;
        *end-- = tmp;
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s INPUT OUTPUT\n", argv[0]);
        return 2;
    }

    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror("fopen input");
        return 1;
    }
    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        perror("fopen output");
        fclose(in);
        return 1;
    }

    char *line = NULL;
    size_t capacity = 0;
    ssize_t length;
    while ((length = getline(&line, &capacity, in)) >= 0) {
        size_t body = (size_t)length;
        if (body > 0 && line[body - 1] == '\n') {
            body--;
        }
        if (body > 1) {
            reverse(line, line + body - 1);
        }
        if (fwrite(line, 1, (size_t)length, out) != (size_t)length) {
            perror("fwrite");
            free(line);
            fclose(in);
            fclose(out);
            return 1;
        }
    }

    free(line);
    fclose(in);
    if (fclose(out) != 0) {
        perror("fclose output");
        return 1;
    }
    return 0;
}
