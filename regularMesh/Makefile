# by R.Lie, Nov 01, 2002

include /usr/local/AFEPack/Make.global_options

source = $(wildcard *.cpp)
object = $(patsubst %.cpp, %.o, $(source))
LDFLAGS += -L/usr/local/AFEPack/library/lib -lAFEPack

all : $(object)
	$(CXX) -o main *.o $(LDFLAGS) $(LIBS)


%.o : %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

step1 : $(object)
	$(CXX) -o $@ step1.o $(LDFLAGS) $(LIBS)

step2 : $(object)
	$(CXX) -o $@ step2.o $(LDFLAGS) $(LIBS)

clean :
	-rm -rf $(object)
	-rm -rf main
	-rm -f *.[nes]
	-rm -f *.dx

.PHONY : default clean
