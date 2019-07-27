# cli.py
import click

@click.command()
@click.option('--gan_type', prompt=True)
def gan():
  pass


@click.command()
@click.argument('location')
def main(location):
    print("I'm a beautiful CLI")
    print(location)
    gan()

if __name__ == "__main__":
    main()
