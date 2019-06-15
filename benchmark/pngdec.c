#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <stdint.h>

void gettime(uint64_t* utime, uint64_t* stime)
{
	struct rusage ru;
	getrusage(RUSAGE_SELF, &ru);
	*utime = ru.ru_utime.tv_usec + ru.ru_utime.tv_sec * 1000 * 1000;
	*stime = ru.ru_stime.tv_usec + ru.ru_stime.tv_sec * 1000 * 1000;
}

int main(int argc, const char **argv)
{
	const char** args = argv + 1;
	const char* file;
	while (file = *args++) {
		int fd;
		struct stat st;
		void* pngdata;
		size_t pngsize;
		uint64_t u0, u1, s0, s1;
		
		fd = open(file, O_RDONLY, 0);
		if (fd == -1) {
			fprintf(stderr, "Failed to open %s. %s\n", file, strerror(errno));
			continue;
		}

		if (fstat(fd, &st) == -1) {
			close(fd);
			continue;
		};

		pngsize = st.st_size;
		pngdata = malloc(pngsize);
		if (!pngdata) {
			close(fd);
			continue;
		}

		read(fd, pngdata, pngsize);

		close(fd);

		png_image image;
		memset(&image, 0, sizeof image);
   		image.version = PNG_IMAGE_VERSION;

		if (png_image_begin_read_from_memory(&image, pngdata, pngsize))  {
			image.format = PNG_FORMAT_RGBA;
			size_t bufsize = PNG_IMAGE_SIZE(image);
			void* buf = malloc(bufsize);

			gettime(&u0, &s0);
			png_image_finish_read(&image, NULL, buf, 0, NULL);
			gettime(&u1, &s1);
			
			printf("%s,%d,%d,%ld,%ld\n", file, image.width, image.height, u1 - u0, s1 - s0);

			free(buf);
		}


		free(pngdata);
	}
		
	
	return 0;
}
