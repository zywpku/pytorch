import os
import argparse
from itertools import count
from ..autograd.utils import CodeTemplate, write
from ..autograd.gen_autograd import load_aten_declarations

template_path = os.path.join(os.path.dirname(__file__), 'templates')

ATEN_DISPATCH_H = CodeTemplate.from_file(template_path + '/aten_dispatch.h')
ATEN_DISPATCH_CPP = CodeTemplate.from_file(template_path + '/aten_dispatch.cpp')

ATTR_METHOD_MAP = {
    'int64_t': 'i',
    'IntList': 'is',
    'Scalar': 't',
    'bool': 'i',
    'double': 'f',
    'std::array<bool,2>': 'is',
    'std::array<bool,3>': 'is',
    'std::array<bool,4>': 'is',
}

TYPE_CASTS = {
    'std::array<bool,2>': 'as_bool_array<2>',
    'std::array<bool,3>': 'as_bool_array<3>',
    'std::array<bool,4>': 'as_bool_array<4>',
    'Scalar': 'Scalar',
    'IntList': 'std::vector<int64_t>',
}

KW_ASSIGNMENT = CodeTemplate("""\
auto ${name} = ${type_cast}(node->${method}(Symbol("${name}")));\
""")

POS_ASSIGNMENT = CodeTemplate("""\
auto ${name} = tensor_as<${type}>(TensorTemporary(inputs[${arg_idx}]).value());\
""")

CALL_NAMESPACE = CodeTemplate("at::${name}(${args})")
CALL_METHOD = CodeTemplate("TensorTemporary(inputs[0]).value().${name}(${args})")

CONSTRUCTOR = CodeTemplate("""\
{"${descriptor}", [](Node *node) {
  ${kw_assignments}
  return TensorOp([=](const list_of_retainable & inputs,
                      list_of_retainable & outputs) {
    autograd::profiler::RecordFunction record("${name}");
    AutoGPU device_guard(deviceForInputs(inputs));
    ${pos_assignments}
    pack_list(outputs, ${call});
  }, "${name}", ${num_inputs});
}},
""")


def is_jit_op(decl):
    return (not decl['api_name'].endswith('_') and
            not decl['name'].endswith('_out') and
            not any(arg['simple_type'] == 'Generator' for arg in decl['arguments']) and
            not any(arg['simple_type'] == 'SparseTensor' for arg in decl['arguments']) and
            not any(arg['simple_type'] == 'Storage' for arg in decl['arguments']) and
            any(arg['simple_type'] in {'Tensor', 'TensorList'} for arg in decl['arguments']) and
            'Tensor' in decl['return_type'])


prefer_tensor_overload = {
    'lt-2', 'gt-2', 'le-2', 'ge-2', 'eq-2', 'ne-2', 'pow-2', 'add-3', 'sub-3',
    'mul-2', 'div-2', 'fmod-2', 'remainder-2'
}


def gen_jit_dispatch(declarations, out):
    aten_decls = load_aten_declarations(declarations)
    jit_decls = [d for d in aten_decls if is_jit_op(d)]

    def is_tensor_arg(arg):
        return arg['simple_type'] in {'Tensor', 'TensorList'}

    for decl in jit_decls:
        arguments = decl['arguments']
        name = decl['name']
        has_tensorlist = any(arg['simple_type'] == 'TensorList' for arg in arguments)
        has_scalar = any(arg['simple_type'] == 'Scalar' for arg in arguments)
        if has_tensorlist:
            continue
        if has_scalar:
            continue

    ops = {}
    for decl in jit_decls:
        arguments = decl['arguments']
        name = decl['name']
        has_tensorlist = any(arg['simple_type'] == 'TensorList' for arg in arguments)

        scalar_arg_idx = [i for i, arg in enumerate(arguments) if not is_tensor_arg(arg)]
        tensor_arg_idx = [i for i, arg in enumerate(arguments) if is_tensor_arg(arg)]

        # Right now, we generate dispatch methods that either take all non-tensor arguments
        # as attributes, or don't use any attributes at all. In the future we might want to
        # have something in the middle too (might be useful for e.g. constant propagation
        # into attributes, as that would allow us to avoid reparsing tensors into scalar
        # args at every invocation).
        # NB: if there are no scalar args then both options on LHS are equivalent, so deduplicate them.
        scalar_arg_idx_iter = ([], scalar_arg_idx) if scalar_arg_idx else ([],)
        for pos_scalar_arg_idx in scalar_arg_idx_iter:
            pos_scalar_args = [arguments[i] for i in pos_scalar_arg_idx]
            kw_scalar_args = [arguments[i] for i in scalar_arg_idx
                              if i not in pos_scalar_arg_idx]
            # Descriptor is a unique identifier for a particular overload of an op
            attr_names = sorted([arg['name'] for arg in kw_scalar_args])
            num_inputs = len(tensor_arg_idx) + len(pos_scalar_arg_idx) if not has_tensorlist else '*'
            descriptor = '-'.join([decl['name'], str(num_inputs)] + attr_names)

            kw_assignments = [KW_ASSIGNMENT.substitute(type_cast=TYPE_CASTS.get(arg['simple_type'], arg['simple_type']),
                                                       name=arg['name'],
                                                       method=ATTR_METHOD_MAP[arg['simple_type']])
                              for arg in kw_scalar_args]
            # TODO: if has_tensor_list then we need to count form the end!
            pos_scalar_arg_offsets = [i + len([None for t_idx in tensor_arg_idx if t_idx < s_idx])
                                      for i, s_idx in enumerate(scalar_arg_idx)]
            pos_assignments = [POS_ASSIGNMENT.substitute(type=arg['simple_type'],
                                                         name=arg['name'],
                                                         arg_idx=arg_idx)
                               for arg_idx, arg in zip(pos_scalar_arg_offsets, pos_scalar_args)]

            # Generate the actuall ATen call. This gets a bit tricky because of
            # TensorList arguments, and functions that are only available as methods.
            if 'namespace' in decl['method_of']:
                if has_tensorlist:
                    if sum(map(is_tensor_arg, arguments)) != 1:
                        # TODO: support this
                        continue
                    args = ['TensorTemporaryList(inputs)' if is_tensor_arg(arg) else arg['name']
                            for arg in arguments]
                else:
                    tensor_id = iter(count(start=0))
                    args = ['TensorTemporary(inputs[{}]).value()'.format(
                        next(tensor_id)) if is_tensor_arg(arg) else arg['name']
                        for arg in arguments]
                call = CALL_NAMESPACE.substitute(name=name, args=args)
            else:
                tensor_id = iter(count(start=1))
                args = ['TensorTemporary(inputs[{}]).value()'.format(next(tensor_id))
                        if is_tensor_arg(arg) else arg['name']
                        for arg in arguments[1:]]
                call = CALL_METHOD.substitute(name=name, args=args)

            constructor = CONSTRUCTOR.substitute(descriptor=descriptor, name=name, call=call,
                                                 kw_assignments=kw_assignments,
                                                 pos_assignments=pos_assignments,
                                                 # num_inputs is only used in AutogradClosure, which
                                                 # is going to be removed soon anyway. There's no good value
                                                 # we can provide for cat.
                                                 num_inputs=num_inputs if num_inputs != "*" else 0)
            if descriptor not in ops:
                ops[descriptor] = (constructor, decl)
            else:
                # There are still some situations where we can recover from an ambiguous descriptor.
                def fail():
                    raise RuntimeError('descriptor conflict: ' + descriptor)

                # If there are two overloads with the same descriptor, that differ only by a type of a
                # single argument, where one of them takes a tensor, while another one takes an
                # at::Scalar as a positional scalar arg, then prefer the tensor overload.
                # It should get broadcasted correctly.
                if descriptor in prefer_tensor_overload:
                    _, prev_decl = ops[descriptor]
                    prev_arguments = prev_decl['arguments']
                    if len(prev_arguments) != len(arguments):
                        fail()
                    arg_diff = [(old['simple_type'], new['simple_type'])
                                for old, new in zip(prev_arguments, arguments)
                                if old['simple_type'] != new['simple_type']]
                    if len(arg_diff) > 1:
                        fail()
                    [(prev_tp, new_tp)] = arg_diff
                    if prev_tp == 'Scalar' and new_tp == 'Tensor':
                        ops[descriptor] = (constructor, decl)
                    elif prev_tp != 'Tensor' or new_tp != 'Scalar':
                        fail()
                else:
                    fail()

    # Sort the generated snippets to ensure that the generation is deterministic
    env = {'constructors': sorted([v[0] for v in ops.values()])}
    write(out, 'aten_dispatch.h', ATEN_DISPATCH_H, env)
    write(out, 'aten_dispatch.cpp', ATEN_DISPATCH_CPP, env)


def main():
    parser = argparse.ArgumentParser(
        description='Generate JIT op dispatch')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    args = parser.parse_args()
    gen_jit_dispatch(args.declarations, args.out)


if __name__ == '__main__':
    main()
