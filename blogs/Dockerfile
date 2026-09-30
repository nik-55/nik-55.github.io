FROM ruby:3.3

WORKDIR /gems
RUN printf 'source "https://rubygems.org"\ngem "github-pages", group: :jekyll_plugins\ngem "webrick"\n' > Gemfile \
    && bundle install
ENV PAGES_REPO_NWO=nik-55/nik-55.github.io

EXPOSE 4000
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--source", "/site", "--destination", "/tmp/_site"]
